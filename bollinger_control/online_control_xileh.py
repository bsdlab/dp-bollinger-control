# Applying SPoC filter to an LSL stream online

import threading
import time
import tomllib
from pathlib import Path

import numpy as np
import pylsl
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
from xileh import xPData, xPipeline

from bollinger_control.utils.logger import logger


def init_pdata(
    config_path: Path = Path("./configs/config.toml"),
) -> xPData:
    """Initialize a pdata container including a data buffer of size nbuffer
    for a total nchannels

    Parameters
    ----------
    nbuffer : int
        number of samples to keep in the FIFO buffer
    nchannels : int
        number of channels to be used in the buffer
    config_path : Path
        path to the config toml

    Returns
    -------
    pdata :  xPData
        the data container used for processing
    """

    pdata = xPData(
        [
            xPData(
                tomllib.load(open(config_path, "rb")),
                name="config",
                header=dict(conf_path=config_path),
            ),
        ],
        name="bollinger_online_control",
        header=dict(tlast_changed=time.time()),
    )

    return pdata


def connect_stream_watcher(pdata: xPData) -> xPData:
    """Connect the stream watchers"""
    sw = StreamWatcher(
        pdata.config.data["stream_to_query"]["stream"],
        buffer_size_s=pdata.config.data["stream_to_query"]["buffer_size_s"],
    )
    sw.connect_to_stream()
    idx = [
        i
        for i, ch in enumerate(sw.channel_names)
        if ch in pdata.config.data["stream_to_query"]["channels"]
    ]
    srate = sw.streams[0].nominal_srate()
    pdata.add(
        sw,
        "stream_watcher",
        header=dict(
            srate=srate,  # for now allow only constant sampling rate for all SWS which are being used.       # noqa
            nchannels=len(idx),  # only the selected are relevant
            selected_ch_idx=idx,
        ),
    )

    # once connected -> translate the horizon time of bollinger to the
    # number of samples depending on the incoming streams srate
    pdata.config.data["bollinger"]["time_horizon_n"] = (
        pdata.config.data["bollinger"]["time_horizon_s"] * srate
    )

    return pdata


def calc_bollinger_bands(pdata: xPData) -> xPData:
    # the stream_watchers buffer should be just 1 channel
    data = pdata.stream_watcher.data.unfold_buffer()

    bollconf = pdata.config.data["bollinger"]
    mean = data[-bollconf["time_horizon_n"] :].mean(axis=0)
    std = data[-bollconf["time_horizon_n"] :].std(axis=0)

    # bollinger lower, value, bollinger upper
    vals = [
        mean - std * bollconf["n_std"],
        data[-1],
        mean + std * bollconf["n_std"],
    ]

    # Add to the outbuffer
    icurr = pdata.outbuffer.header["icurr"]
    pdata.outbuffer.data[icurr, :] = vals
    pdata.outbuffer.header["icurr"] = (
        icurr + 1 if icurr + 1 < pdata.outbuffer.data.shape[0] else 0
    )
    pdata.outbuffer.header["nnew"] += 1

    return pdata


def init_lsl_outlet(pdata: xPData) -> xPData:
    cfg = pdata.config.data["lsl_outlet"]
    bufferlen = (
        pdata.config.data["outbuffer"]["size_s"]
        * pdata.stream_watcher.header["srate"]
    )
    n_channels = 3

    info = pylsl.StreamInfo(
        cfg["name"],
        cfg["type"],
        n_channels,
        pdata.config.data["lsl_outlet"]["nominal_freq_hz"],
        cfg["format"],
    )

    # enrich a channel name
    chns = info.desc().append_child("channels")
    for chn in ["min", "decoded_value", "max"]:
        ch = chns.append_child("channel")
        ch.append_child_value("label", f"bollinger_control_{chn}")
        ch.append_child_value("unit", "AU")
        ch.append_child_value("type", "bollinger_control")
        ch.append_child_value("scaling_factor", "1")

    pdata.add(pylsl.StreamOutlet(info), name="outlet")

    # using an outputbuffer for simplicity
    pdata.add(
        np.empty((bufferlen, n_channels)),
        name="outbuffer",
    )

    return pdata


def query_stream_watcher(pdata: xPData) -> xPData:
    pdata.stream_watcher.data.update()
    return pdata


def stream_result(pdata: xPData):
    # 6.38 µs ± 5.22 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each
    icurr = pdata.outbuffer.header["icurr"]
    nnew = pdata.outbuffer.header["nnew"]
    outbuffer = pdata.outbuffer.data

    if icurr == 0:  # icurr was just set to 0
        data = outbuffer[-nnew:]
    elif icurr < nnew:  # last addition was overflown
        data = np.hstack([outbuffer[-(nnew - icurr) :], outbuffer[:icurr]])
    else:
        data = outbuffer[icurr - nnew : icurr]

    data = data.reshape(-1, 1)
    logger.debug(f"Streaming data: {data}")
    pdata.outlet.data.push_chunk(data.tolist())

    pdata.outbuffer.header["nnew"] = 0

    return pdata


def process_loop(
    pdata: xPData,
    pl: xPipeline,
    stop_event: threading.Event = threading.Event(),
):
    """Process the given pipeline in a loop with a given freq"""

    freq_hz = pdata.config.data["lsl_outlet"]["nominal_freq_hz"]
    dt = 1 / freq_hz
    tlast = time.time_ns()

    # wait 2s for to get a full buffer to start from
    time.sleep(2)
    pdata.stream_watcher.data.update()

    while not stop_event.is_set():
        now = time.time_ns()
        if now - tlast > dt * 10**9:
            pl.eval(pdata)
            tlast = now


def interpret_control(pdata: xPData) -> xPData:
    lower, val, upper = pdata.outbuffer.data[-1, :]
    tlast_changed = pdata.header["tlast_changed"]
    dgrace = time.time() - tlast_changed

    if dgrace > pdata.config.data["bollinger"]["grace_s"]:
        if (
            val < lower
        ):  # Assuming that a higher value is closer to the on state
            logger.debug("CONTROL WOULD SEND STIM ON")
        elif val > higher:
            logger.debug("CONTROL WOULD SEND STIM OFF")
    else:
        logger.debug("GRACE PERIOD NOT OVER YET")

    return pdata


# ---------------- Pipeline ---------------------------------------------------
online_control_pl = xPipeline("online_filter_pl", silent=True)
online_control_pl.add_steps(
    ("query_stream_watchers", query_stream_watcher),
    ("calc_bollinger_bands", calc_bollinger_bands),
    ("stream_result", stream_result),
)

setup_pl = xPipeline("setup_pl", silent=True)
setup_pl.add_steps(
    ("connect_stream_watcher", connect_stream_watcher),
    ("init_lsl_outlet", init_lsl_outlet),
)


if __name__ == "__main__":
    pdata = init_pdata()

    pdata.lsl_config.data["stream_to_query"]["streams"] = ["mock_EEG_stream"]

    setup_pl.eval(pdata)

    logger.setLevel(10)
    pdata.add([], name="tdata")  # to store the transformed data for testing
    pdata.add([], name="tdata_raw")  # to store the raw data for testing
    pdata.add([], name="tdata_samples")  # to store the raw data for testing
    process_loop(pdata, online_control_pl)

import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pylsl
from dareplane_utils.default_server.server import DefaultServer
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
from fire import Fire

from bollinger_control.gate_keeper import GateKeeper
from bollinger_control.online_control import Context, process_loop
from bollinger_control.utils.logging import logger


def init_lsl_outlet(cfg: dict) -> pylsl.StreamOutlet:
    """Initialize the LSL outlet"""

    cfg_out = cfg["lsl_outlet"]
    n_channels = 3

    info = pylsl.StreamInfo(
        cfg_out["name"],
        cfg_out["type"],
        n_channels,
        cfg_out["nominal_freq_hz"],
        cfg_out["format"],
    )

    # enrich a channel name
    chns = info.desc().append_child("channels")
    for chn in ["min", "decoded_value", "max"]:
        ch = chns.append_child("channel")
        ch.append_child_value("label", f"bollinger_control_{chn}")
        ch.append_child_value("unit", "AU")
        ch.append_child_value("type", "bollinger_control")
        ch.append_child_value("scaling_factor", "1")

    outlet = pylsl.StreamOutlet(info)
    return outlet


def init_marker_outler() -> pylsl.StreamOutlet:

    stream_info = pylsl.StreamInfo(
        name="BollingerControlMarkerStream",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id="BollingerControlMarker",
    )

    stream_outlet = pylsl.StreamOutlet(stream_info)

    return stream_outlet


def run_bollinger_control(
    server: DefaultServer | None = None,
    config_path: Path = Path("./configs/config.toml"),
) -> tuple[threading.Thread, threading.Event]:
    """

    Parameters
    ----------
    server : DefaultServer | None
        the server used to run this module, used or control room callbacks
        triggering the stimulation

    config_path : Path
        path to the configuration file

    Returns
    -------
    tuple[threading.Thread, threading.Event]
        the thread and the stop event to control the thread

    """
    # TODO: Consider just passing the config and the server to the thread,
    #       then initiliaze the StreamWatcher, GateKeeper and LSL outlet in
    #       the thread.
    cfg = tomllib.load(open(config_path, "rb"))

    sw = StreamWatcher(
        cfg["stream_to_query"]["stream"],
        buffer_size_s=cfg["stream_to_query"]["buffer_size_s"],
    )
    sw.connect_to_stream()

    if cfg["lsl_outlet"]["nominal_freq_hz"] == "derive":
        cfg["lsl_outlet"]["nominal_freq_hz"] = sw.streams[0].nominal_srate()

    outlet = init_lsl_outlet(cfg)
    marker_outlet = init_marker_outler()
    gatekeeper = GateKeeper(**cfg["stim"]["gatekeeper"])

    # setup_pl.eval(pdata)
    stop_event = threading.Event()
    stop_event.clear()
    ctx = Context(
        sw,
        gatekeeper,
        outlet,
        marker_outlet,
        server,
        cfg=cfg,
        stop_event=stop_event,
    )

    thread = threading.Thread(target=process_loop, args=(ctx,))

    logger.debug(f"Created {thread=}")
    thread.start()

    return thread, stop_event


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    Fire(run_bollinger_control)

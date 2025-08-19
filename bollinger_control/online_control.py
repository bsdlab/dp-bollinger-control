import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pylsl
from dareplane_utils.default_server.server import DefaultServer
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher

from bollinger_control.gate_keeper import GateKeeper
from bollinger_control.utils.logging import logger


@dataclass
class Context:
    sw: StreamWatcher
    gk: GateKeeper
    outlet: pylsl.StreamOutlet
    marker_outlet: pylsl.StreamOutlet
    server: DefaultServer | None
    stop_event: threading.Event = threading.Event()
    cfg: list = field(default_factory=dict)


def update_control_buffer(
    sw: StreamWatcher,
    control_buffer: np.ndarray,
    time_horizon_n: int,
    n_std: float,
    curri: int,
    idx_signal: int = 0,
) -> int:
    data = sw.unfold_buffer()[:, -1:]

    # Use pandas rolling function for speed -> conversion for n=500 input buffer
    # takes about 7us on M1
    df = pd.DataFrame(data)
    mean = df.iloc[:, idx_signal].rolling(time_horizon_n).mean()
    std = df.iloc[:, idx_signal].rolling(time_horizon_n).std()
    # logger.debug(f"Derived from buffer data: {mean=}, {std=}")
    df["upper"] = mean + std * n_std
    df["lower"] = mean - std * n_std
    df["mean"] = mean

    if sw.n_new == 0:
        # logger.debug(f"{sw.n_new=} - no new data")
        return 0

    new = min(control_buffer.shape[0] - curri, sw.n_new, data.shape[0])

    control_buffer[curri : curri + new] = df[["upper", idx_signal, "lower", "mean"]][
        -new:
    ]

    return new


def derive_stim_state(buffer: np.ndarray) -> str:
    """Derive control from the control buffer"""
    ctrl = np.asarray([np.nan] * len(buffer))

    # signal < lower limit --> classifier output for LDA label with >0 == stim ON
    ctrl[buffer[:, 1] < buffer[:, 2]] = 1

    # signal > upper limit
    ctrl[buffer[:, 1] > buffer[:, 0]] = 0

    if all(np.isnan(ctrl)):
        return "no_change"
    else:
        # take the last value of currently processed buffer
        last_v = ctrl[~np.isnan(ctrl)][-1]

    # logger.debug(f"Derived control state: {last_v=}, {ctrl[-5:]=}")

    condition_map = {1: "on", 0: "off"}

    return condition_map[last_v]


def create_control_cmd(stim: str, cfg: dict) -> str:
    ctr_func_map = {
        "AO": create_control_cmd_ao,
    }

    return ctr_func_map[cfg["stim"]["gatekeeper"]["stimulator"]](stim, cfg)


def create_control_cmd_ao(stim: str, cfg: dict) -> str:
    """CallbackBroker requires messages of the format:
    <target_module_name>|<PCOMM>|{payload}"""

    scfg = cfg["stim"]["stim_on"]
    if stim == "off":
        payload = dict(StimChannel=scfg["stim_channel"])
        cmd = f"{cfg['stim']['dp_module_name']}|STOPSTIM|{payload}"
        cmd = cmd.replace("'", '"')
        return cmd
    elif stim == "on":
        # Take the values from the config
        # logger.debug(f"Creating stim command with {scfg=}")
        payload = dict(
            StimChannel=scfg["stim_channel"],
            FirstPhaseDelay_mS=scfg["first_phase_delay_ms"],
            FirstPhaseAmpl_mA=scfg["first_phase_ampl_mA"],
            FirstPhaseWidth_mS=scfg["first_phase_width_ms"],
            SecondPhaseDelay_mS=scfg["second_phase_delay_ms"],
            SecondPhaseAmpl_mA=scfg["second_phase_ampl_mA"],
            SecondPhaseWidth_mS=scfg["second_phase_width_ms"],
            Freq_hZ=scfg["freq_hz"],
            Duration_sec=scfg["duration_s"],
            ReturnChannel=scfg["return_channel"],
        )

        cmd = f"{cfg['stim']['dp_module_name']}|STARTSTIM|{payload}"

        # ensure double quotes for valid json
        cmd = cmd.replace("'", '"')

        return cmd

    else:
        logger.error(f"UNKNOWN {stim=} value")
        raise ValueError(f"UNKNOWN {stim=} value")


def process_loop(
    ctx: Context,
):
    """Process the given pipeline in a loop with a given freq"""
    cfg = ctx.cfg
    outlet = ctx.outlet
    gatekeeper = ctx.gk

    freq_hz = outlet.get_info().nominal_srate()
    dt = 1 / freq_hz

    # Note: this buffer will always be filled from 0 onwards until the data
    # if dumped to the outlet, the buffer used for the calculation of the
    # moving averages stems will be the StreamWatchers
    control_buffer = np.zeros((int(cfg["outbuffer"]["size_s"] * freq_hz), 4))
    curri = 0

    time_horizon_n = int(cfg["bollinger"]["time_horizon_s"] * freq_hz)
    n_std = cfg["bollinger"]["n_std"]

    control_stim = False

    # sent_samples = 0
    start_time = pylsl.local_clock()
    last_update = pylsl.local_clock()

    while not ctx.stop_event.is_set():
        dt_s = pylsl.local_clock() - last_update

        if dt_s > dt:
            # update stream watcher
            ctx.sw.update()
            last_update = pylsl.local_clock()

            added = update_control_buffer(
                ctx.sw, control_buffer, time_horizon_n, n_std, curri
            )
            curri += added

            if control_stim:
                # interpret stimulation
                # logger.info(f"{added=}, {control_buffer[:1]}")
                stim = derive_stim_state(control_buffer[:curri])

                if stim != "no_change":
                    # currently hardwire the stim command
                    cmd = create_control_cmd(stim, cfg)

                    # validate
                    valid_cmd = gatekeeper.validate_stim_command(cmd)
                    # logger.debug(f"Validation of {cmd=} --> {valid_cmd=}")
                    if valid_cmd:
                        gatekeeper.stim_state = stim

                        # send control back to control room which in turn triggers
                        # the control module
                        if ctx.server is not None:
                            logger.debug("Sending cmd to control room")
                            ctx.server.current_conn.sendall(cmd.encode())
                            ctx.marker_outlet.push_sample([cmd])

            else:
                # Time to wait to fill up buffer
                if (pylsl.local_clock() - start_time) > cfg["stim"]["initial_delay_s"]:
                    control_stim = True

            # send to lsl
            chunk = control_buffer[:curri, :]
            for smp in chunk:
                outlet.push_sample(smp)
            ctx.sw.n_new = 0
            curri = 0
        else:
            # keeping the clock more simple for better resource usage
            time.sleep(dt)
            # tsleep = 0.8 * (dt - dt_s)
            # sleep_s(tsleep)

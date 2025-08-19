# Implementation of a gatekeeper that checks for valid stimulation commands / e.g. tracking current stim state and enforcing grace period
import json
import time
from dataclasses import dataclass, field

from bollinger_control.utils.logging import logger


@dataclass
class GateKeeper:
    stim_state: str = "off"
    last_stim_command_time_ns: float = time.perf_counter_ns()
    stimulator: str = "not_specified"
    max_amp_mA: float = 0
    max_width_ms: float = 0
    freq_range: list = field(default_factory=list)
    black_list_freq: list = field(default_factory=list)
    white_list_contacts: list = field(default_factory=list)
    grace_period_s: float = 0

    def validate_stim_command(self, cmd: str) -> bool:
        validation_func_map = {
            "AO": self.validate_ao_stim_command,
        }
        # Assume the command string is of structure
        # <target_module_name>|<PCOMM>|{payload}"""
        #
        payload = json.loads(cmd.split("|")[2])
        pcomm = cmd.split("|")[1]

        return validation_func_map[self.stimulator](pcomm, payload)

    def validate_ao_stim_command(self, pcomm: str, config: dict) -> bool:
        # Check command against current state -> only continue if command would
        # induce a change
        if pcomm == "STARTSTIM" and self.stim_state == "on":
            # logger.debug("Stimulation already on - ignoring command")
            return False
        elif pcomm == "STOPSTIM" and self.stim_state == "off":
            # logger.debug("Stimulation already off - ignoring command")
            return False

        # Turning off only requires grace period check
        if pcomm == "STOPSTIM":
            gp_passed = self.check_grace_period()

            # Reset grace period, as the command will be used
            if gp_passed:
                self.last_stim_command_time_ns = time.perf_counter_ns()

            return gp_passed

        checks = []

        # If time since last valid command is not larger than grace period
        # returning invalid
        checks.append(self.check_grace_period())

        # check stim channel in white list
        checks.append(self.check_stim_channel_in_white_list(config["StimChannel"]))

        # Check that stim command is charge balanced
        checks.append(
            self.check_charge_balance(
                config["FirstPhaseAmpl_mA"],
                config["FirstPhaseWidth_mS"],
                config["SecondPhaseAmpl_mA"],
                config["SecondPhaseWidth_mS"],
            )
        )

        # Check that amplitudes to not exceed max_amp
        #    amplitude 1
        checks.append(self.check_amplitude(config["FirstPhaseAmpl_mA"]))
        #    amplitude 2
        checks.append(self.check_amplitude(config["SecondPhaseAmpl_mA"]))

        # Check that width is in limit
        checks.append(self.check_stimulation_width(config["FirstPhaseWidth_mS"]))

        checks.append(self.check_stimulation_width(config["SecondPhaseWidth_mS"]))

        # check valid frequency
        checks.append(self.check_frequency_in_admissible_range(config["Freq_hZ"]))

        # all checks valid
        if all(checks):
            # logger.debug("Valid stimulation command - resetting grace period")
            self.last_stim_command_time_ns = time.perf_counter_ns()
            return True

        else:
            tests = [
                "grace_period",
                "stim_channel",
                "charge_balance",
                "amp1",
                "amp2",
                "width1",
                "width2",
                "freq",
            ]
            logger.debug(f"Validation failure: {dict(zip(tests, checks))}")
            return False

    def validate_cortec_stim_command(self, config: dict) -> bool:
        raise NotImplementedError

    def check_grace_period(self) -> bool:
        dt = (time.perf_counter_ns() - self.last_stim_command_time_ns) * 1e-9
        # logger.debug(
        #     f"Time since last command: {dt} - grace_period: {self.grace_period_s}"
        # )

        return dt > self.grace_period_s

    def check_charge_balance(self, a1: float, w1: float, a2: float, w2: float) -> bool:
        return a1 * w1 + a2 * w2 == 0

    def check_stim_channel_in_white_list(self, channel: str | int) -> bool:
        if isinstance(channel, int):
            channel = str(channel)

        return channel in self.white_list_contacts

    def check_amplitude(self, amp: float) -> bool:
        return amp <= self.max_amp_mA

    def check_frequency_in_admissible_range(self, freq: float) -> bool:
        return freq >= self.freq_range[0] and freq <= self.freq_range[1]

    def check_stimulation_width(self, width: float) -> bool:
        return width <= self.max_width_ms


## For AO stimulator with our dp-ao-control / dareplane-ao-communication module
# /*
# * The Following functions are available:
# *      STARTREC()
# *      STOPREC()
# *      STARTSTIM(
# *                              parameter1: StimChannel,
# *                              parameter2: FirstPhaseDelay_mS,
# *                              parameter3: FirstPhaseAmpl_mA,
# *                              parameter4: FirstPhaseWidth_mS,
# *                              parameter5: SecondPhaseDelay_mS,
# *                              parameter6: SecondPhaseAmpl_mA,
# *                              parameter7: SecondPhaseWidth_mS,
# *                              parameter8: Freq_hZ,
# *                              parameter9: Duration_sec,
# *                              parameter10: ReturnChannel
# *      )
# *      STOPSTIM(parameter1: StimChannel)
# *      SETPATH(parameter1: Path)
# *
# *      e.g.:
# *      STOPSTIM|10287
# */

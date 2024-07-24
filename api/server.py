from functools import partial

from dareplane_utils.default_server.server import DefaultServer
from fire import Fire

from bollinger_control.main import run_bollinger_control
from bollinger_control.utils.logging import logger


def main(port: int = 8080, ip: str = "127.0.0.1", loglevel: int = 10):
    logger.setLevel(loglevel)
    server = DefaultServer(
        port, ip=ip, pcommand_map={}, name="bollinger_control_server"
    )

    # Thread needs access to the server for sending back
    pcommand_map = {
        "STARTCONTROL": partial(run_bollinger_control, server=server),
    }

    server.pcommand_map.update(pcommand_map)

    # initialize to start the socket
    server.init_server()
    # start processing of the server
    server.start_listening()

    return 0


if __name__ == "__main__":
    Fire(main)

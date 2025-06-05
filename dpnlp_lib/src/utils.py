import logging


def enable_info_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


enable_info_logs()
logger = logging.getLogger(__name__)

# TODO: some functions for plotting, incremental checkpointing, saving log outputs to respective folders, hardware monitoring, communication rounds

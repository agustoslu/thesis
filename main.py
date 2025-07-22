import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

from dpnlp_lib.src.utils import set_global_seeds, generate_run_uuid
from dpnlp_lib.src.builder import run_builder
from dpnlp_lib.src.server import run_flower_server


import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_global_seeds(cfg.builder.seed)
    run_uuid = generate_run_uuid()
    logger.info("Starting data building process...")
    client_train, client_test = run_builder(cfg, run_uuid)
    logger.info(
        "Data building complete. Splits: %s",
        {"client_train": (len(client_train)), "client_test": (len(client_test))},
    )
    logger.info("Client train partitions: %s", client_train)
    logger.info("Client test partitions: %s", client_test)
    logger.info("Starting federated training process...")

    logger.info(f"Instantiating task: {cfg.task._target_}")

    logger.info(cfg.task)
    logger.info(f"Model_config: {cfg.task.model_config}")
    task_obj = get_class(cfg.task._target_)()
    logger.info(f"Instantiating strategy: {cfg.algorithm._target_}")
    strategy_obj = instantiate(
        cfg.algorithm,
        task=task_obj,
        model_config=cfg.task.model_config,
        run_uuid=run_uuid,
    )

    logger.info("Starting Flower server...")
    run_flower_server(cfg, strategy_obj, task_obj, client_train, client_test, run_uuid)
    logger.info("Federated training completed!")


if __name__ == "__main__":
    main()

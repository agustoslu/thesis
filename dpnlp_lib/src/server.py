import torch
from typing import Any
import opacus

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr.common import Context
from flwr.client import ClientApp
from flwr.server.strategy import Strategy, FedAvg
from dpnlp_lib.src.client import get_client_fn
from dpnlp_lib.src.dp_sgd import DPSGDFedAvg
# from dpnlp_lib.src.dp_ftrl import DPFTRLFedAvg

from omegaconf import DictConfig, OmegaConf


def server_fn(context: Context, num_rounds: int, strategy_obj) -> ServerAppComponents:
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(config=config, strategy=strategy_obj)


def run_flower_server(
    cfg: DictConfig,
    strategy_obj: FedAvg,
    task_obj,
    train_partitions,
    test_partitions,
    run_uuid,
) -> None:
    OmegaConf.set_struct(cfg, False)
    client_fn = get_client_fn(
        cfg, task_obj, train_partitions, test_partitions, run_uuid
    )
    client_app = ClientApp(client_fn=client_fn)
    num_rounds = cfg.server.num_rounds
    num_supernodes = cfg.builder.num_hospitals
    backend_config = OmegaConf.to_container(cfg.server.backend_config, resolve=True)
    if (
        torch.cuda.is_available()
        and backend_config.get("client_resources", {}).get("num_gpus", 0) == 0
    ):
        backend_config["client_resources"]["num_gpus"] = 1

    server = ServerApp(
        server_fn=lambda context: server_fn(context, num_rounds, strategy_obj)
    )

    run_simulation(
        server_app=server,
        client_app=client_app,
        num_supernodes=num_supernodes,
        backend_config=backend_config,
    )

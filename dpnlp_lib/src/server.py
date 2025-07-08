import torch
from typing import Any
import opacus
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr.common import Context
from dpnlp_lib.src.dp_sgd import DPSGDFedAvg
from dpnlp_lib.src.dp_ftrl import DPFTRLFedAvg
from client import client


def get_strategy(strategy_name: str):
    if strategy_name == "dp_sgd":
        return DPSGDFedAvg()
    elif strategy_name == "dp_ftrl":
        return DPFTRLFedAvg()
    else:
        raise ValueError(f"Unknown: {strategy_name}")


def server_fn(
    context: Context, num_rounds: int, strategy_name: str
) -> ServerAppComponents:
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(config=config, strategy=get_strategy(strategy_name))


def run_flower_server(cfg, **kwargs: Any) -> None:
    backend_config = cfg.server.backend_config  # without it flower defaults to CPU
    if (
        torch.cuda.is_available()
        and backend_config.get("client_resources", {}).get("num_gpus", 0) == 0
    ):
        backend_config["client_resources"]["num_gpus"] = 1

    server = ServerApp(
        server_fn=lambda context: server_fn(
            context, cfg.server.num_rounds, cfg.server.strategy
        )
    )
    run_simulation(
        server_app=server,
        client_app=client,
        **backend_config,
        **kwargs,
    )

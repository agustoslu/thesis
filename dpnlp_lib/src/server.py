import torch
from typing import Any
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr.common import Context
from dp_sgd import DPSGDFedAvg
from dp_ftrl import DPFTRLFedAvg


def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(
        num_rounds=200
    )  # TODO: we might not explore this hyperparameter due to time constraints, still better to switch to config when they are all ready
    return ServerAppComponents(
        config=config, strategy=DPSGDFedAvg()
    )  # pass config with conditional logic to use DPFTRLFedAvg


server = ServerApp(server_fn=server_fn)

backend_config = {"client_resources": None}
if torch.cuda.is_available():
    backend_config = {"client_resources": {"num_gpus": 1}}


def run_flower_server(**kwargs: Any) -> None:
    run_simulation(
        server_app=server,
        client_app=client,
        **kwargs,
    )

import torch
from typing import Any
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr.common import Context
from dpnlp_lib.src.dp_sgd import DPSGDFedAvg
from dpnlp_lib.src.dp_ftrl import DPFTRLFedAvg
from client import client 


def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(
        num_rounds=200
    )
    return ServerAppComponents(
        config=config, strategy=DPSGDFedAvg()
    )  


server = ServerApp(server_fn=server_fn)

backend_config = {"client_resources": None} # needs this additional config otherwise flower will default to CPU
if torch.cuda.is_available():
    backend_config = {"client_resources": {"num_gpus": 1}}


def run_flower_server(**kwargs: Any) -> None:
    run_simulation(
        server_app=server,
        client_app=client,
        **kwargs,
    )

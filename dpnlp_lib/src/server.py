import torch
from typing import Any
from flwr.server import ServerApp, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.typing import NDArray
from flwr.server.strategy import FedAvg
from utils import (
    load_config,
    load_model,
    get_weights,
    freeze_layers,
)


def setup_fedavg_strategy(model: torch.nn.Module, fraction_fit: float) -> FedAvg:
    initial_weights: list[NDArray] = get_weights(model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
    )
    return strategy


def server_fn(context: Context) -> ServerAppComponents:
    config: dict[str, Any] = context.fl_config
    num_rounds: int = config["num_server_rounds"]
    fraction_fit: float = config["fraction_fit"]

    model_id: str = config["model_id"]
    model: torch.nn.Module = load_model(model_id, load_config(model_id))
    model = freeze_layers(model, config["use_lora"])

    strategy: FedAvg = setup_fedavg_strategy(model, fraction_fit)

    server_config = ServerAppComponents(
        strategy=strategy, config={"num_rounds": num_rounds}
    )

    return server_config


app = ServerApp(server_fn=server_fn)

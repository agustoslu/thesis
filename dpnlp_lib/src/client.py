import torch
import numpy as np
from typing import Any, Dict, Tuple
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.typing import NDArray
from utils import (
    load_config,
    load_data,
    load_model,
    create_tokenizer,
    set_weights,
    get_weights,
    freeze_layers,
    train,
    evaluate,
    initialize_privacy_engine,
    compute_delta,
)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        local_epochs: int,
        privacy_mode: bool,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.local_epochs = local_epochs
        self.privacy_mode = privacy_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(
        self, params: list[NDArray], config: Dict[str, Any]
    ) -> Tuple[list[NDArray], int, Dict[str, Any]]:
        set_weights(self.model, params)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, eps=1e-8)

        delta = compute_delta(self.train_dataloader)

        self.model, optimizer, self.train_dataloader, criterion, privacy_engine = (
            initialize_privacy_engine(
                self.model, optimizer, self.train_dataloader, self.privacy_mode, delta
            )
        )

        train(
            self.model,
            self.train_dataloader,
            self.test_dataloader,
            optimizer,
            self.device,
            privacy_engine,
            criterion,
            delta=delta,
            epochs=self.local_epochs,
            log_interval=config["logging_interval"],
            max_batch_size=config["max_physical_batch_size"],
        )

        return get_weights(self.model), len(self.train_dataloader), {}

    def evaluate(
        self, params: list[NDArray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        set_weights(self.model, params)
        loss, accuracy = evaluate(self.model, self.test_dataloader, self.device)
        return float(loss), len(self.test_dataloader), {"accuracy": accuracy}


def client_fn(context: Context) -> FlowerClient:
    config: Dict[str, Any] = context.fl_config
    model_id: str = config["model_id"]

    tokenizer = create_tokenizer(model_id)
    model: torch.nn.Module = load_model(model_id, load_config(model_id))
    model = freeze_layers(model, config["use_lora"])

    train_dataloader, test_dataloader = load_data(
        tokenizer, config["batch_size"], max_seq_length=128
    )

    return FlowerClient(
        model,
        train_dataloader,
        test_dataloader,
        config["local_epochs"],
        config["vanilla"],
    ).to_client()


app = ClientApp(client_fn=client_fn)

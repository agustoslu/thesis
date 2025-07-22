import logging
import torch
import torch.nn as nn
from typing import Any, List, Tuple, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict
import numpy as np
import pandas as pd
import os
import json
import csv
import opacus
import random
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import pickle
import uuid


def enable_info_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


enable_info_logs()
logger = logging.getLogger(__name__)


def collate_fn(
    batch: List[Tuple[Any, Any]],
) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, dim=0)
    return padded_features, labels


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    # from: https://github.com/adap/flower/blob/main/framework/docs/source/tutorial-series-build-a-strategy-from-scratch-pytorch.ipynb
    return [value.cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    # from: https://github.com/adap/flower/blob/main/framework/docs/source/tutorial-series-build-a-strategy-from-scratch-pytorch.ipynb
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_target_delta(data_size: int) -> float:
    # from: https://github.com/matturche/flower_opacus_example/blob/main/flower_helpers.py#L143
    """Generate target delta (δ) based on the dataset size. In (ε, δ)-differential privacy,
    delta is a small number that reflects the (very low) chance that privacy guarantees might not hold.
    A common rule of thumb is to make it smaller than the inverse of the dataset size."""
    denominator = 1
    while data_size // denominator >= 1:
        denominator *= 10
    return 1 / denominator


def wrap_model_with_dp(model: nn.Module) -> nn.Module:
    """
    Wraps the model to make it compatible with PrivacyEngine of Opacus.
    """
    if not ModuleValidator.is_valid(model):
        logger.info("Fixing model for Opacus compatibility...")
        model = ModuleValidator.fix(model)
        assert ModuleValidator.is_valid(model)
        return model


def set_global_seeds(seed: int):
    """Sets the global random seed for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global random seed set to {seed} for the current experiment.")


def generate_run_uuid() -> str:
    uid = str(uuid.uuid4())
    logger.info("Run ID: %s", uid)
    return uid


def get_metrics_dir(run_uuid: str) -> str:
    metrics_dir = os.path.join(os.getcwd(), "metrics", run_uuid)
    os.makedirs(metrics_dir, exist_ok=True)
    return metrics_dir


def get_accountant_dir(run_uuid: str) -> str:
    accountant_dir = os.path.join(os.getcwd(), "accountant", run_uuid)
    os.makedirs(accountant_dir, exist_ok=True)
    return accountant_dir


def get_accountant_state_path(cid: str, run_uuid: str) -> str:
    accountant_dir = get_accountant_dir(run_uuid)
    return os.path.join(accountant_dir, f"accountant_state_{cid}_{run_uuid}.pkl")


def load_accountant_state(privacy_engine, cid: str, run_uuid: str, logger=None) -> None:
    state_path = get_accountant_state_path(cid, run_uuid)
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            accountant_state = pickle.load(f)
        privacy_engine.accountant.load_state_dict(accountant_state)
        if logger:
            logger.info(f"[Client {cid}] Loaded accountant state from {state_path}")


def save_accountant_state(privacy_engine, cid: str, run_uuid: str, logger=None) -> None:
    state_path = get_accountant_state_path(cid, run_uuid)
    accountant_state = privacy_engine.accountant.state_dict()
    with open(state_path, "wb") as f:
        pickle.dump(accountant_state, f)
    if logger:
        logger.info(f"[Client {cid}] Saved accountant state to {state_path}")


def save_metrics(metrics: dict, file_name: str, run_uuid: str) -> None:
    metrics_dir = get_metrics_dir(run_uuid)
    metrics_file = os.path.join(metrics_dir, file_name)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_file}")


def append_row_to_csv(row: dict, file_name: str, run_uuid: str) -> None:
    metrics_dir = get_metrics_dir(run_uuid)
    csv_file = os.path.join(metrics_dir, file_name)
    df = pd.DataFrame([row])
    header = not os.path.exists(csv_file)
    df.to_csv(csv_file, mode="a", header=header, index=False)
    logger.info(f"Row appended to {csv_file}")


class MIMICTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=63,
        num_classes=2,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        out = x[:, -1, :]
        logits = self.classifier(out)
        return logits

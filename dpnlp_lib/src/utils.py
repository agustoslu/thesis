import logging
import torch
import torch.nn as nn
from typing import Any, List, Tuple, Dict
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict
import numpy as np


# TODO: some functions for plotting, incremental checkpointing, saving log outputs to respective folders, hardware monitoring, communication rounds


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
    lengths = [f.shape[0] for f in features]
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, dim=0)
    return padded_features, lengths, labels


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


class MIMICGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=6, dropout=0.3):
        super(MIMICGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.hidden_size = hidden_dim

    def forward(self, padded_sequences, lengths):
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            padded_sequences, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed_sequences)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden

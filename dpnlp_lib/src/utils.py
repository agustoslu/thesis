import logging
import torch
import torch.nn as nn

# TODO: some functions for plotting, incremental checkpointing, saving log outputs to respective folders, hardware monitoring, communication rounds


def enable_info_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


enable_info_logs()
logger = logging.getLogger(__name__)


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

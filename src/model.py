"""A small, unidirectional LSTM with padding-aware final state."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class CategoryLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # h_n contains the state at each sample's last REAL click, not at padding.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.classifier(h_n[-1])

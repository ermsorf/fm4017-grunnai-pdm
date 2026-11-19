"""Backward-compatible imports for the shared RNN data loader."""

from src.data_loader import RNNData, RNNSplit, load_rnn_data, load_rnn_evaluation_data

__all__ = [
    "RNNData",
    "RNNSplit",
    "load_rnn_data",
    "load_rnn_evaluation_data",
]

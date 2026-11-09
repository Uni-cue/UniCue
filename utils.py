import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml

DEFAULT_ARGS = {
    'outputpath': 'experiments',
    'batch_size': 4,
    'num_workers': 6, # Number of dataset loaders
    'mixer_args': {},
    'n_saved': 1,
    'epochs': 400, 
    'optimizer': 'Adam',
    'optimizer_args': {
        'lr': 0.001,
    },
}


class half_lr_rate(object):
    def __init__(self, optimizer, param_name, score_function, patience = 3, min_delta = 0.0):
        self.optimizer = optimizer
        self.param_name = param_name
        self.score_function = score_function
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.min_delta = min_delta

    def __call__(self, evaluator):
        score = self.score_function(evaluator)

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group[self.param_name] = param_group[self.param_name] / 2.0
        else:
            self.best_score = score
            self.counter = 0


def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    arguments = dict(yaml_config, **kwargs)
    # In case some arguments were not passed, replace with default ones
    for key, value in DEFAULT_ARGS.items():
        arguments.setdefault(key, value)
    return arguments


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for tensors of shape ``[T, B, C]``.

    Mirrors the SpeechBrain implementation so the SepFormer-style blocks in
    :mod:`unicine.model` do not depend on SpeechBrain internals.
    """

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [T, B, C]
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PositionalEncodingPermute2D(nn.Module):
    """2-D sinusoidal positional encoding for dual-path tensors ``[B, N, K, S]``.

    The first half of the channels encodes the position along the chunk axis (K)
    and the second half encodes the position along the chunk-index axis (S).
    """

    def __init__(self, channels):
        super().__init__()
        assert channels % 2 == 0
        self.channels = channels
        half = channels // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        B, N, K, S = x.shape
        dtype = self.inv_freq.dtype
        pos_k = torch.arange(K, device=x.device, dtype=dtype)
        pos_s = torch.arange(S, device=x.device, dtype=dtype)
        sin_k = torch.einsum("i,j->ij", pos_k, self.inv_freq)  # [K, half/2]
        sin_s = torch.einsum("i,j->ij", pos_s, self.inv_freq)  # [S, half/2]
        emb_k = torch.cat((sin_k.sin(), sin_k.cos()), dim=-1)  # [K, half]
        emb_s = torch.cat((sin_s.sin(), sin_s.cos()), dim=-1)  # [S, half]
        pe = torch.zeros(K, S, self.channels, device=x.device, dtype=x.dtype)
        pe[:, :, : self.channels // 2] = emb_k.unsqueeze(1)  # broadcast over S
        pe[:, :, self.channels // 2 :] = emb_s.unsqueeze(0)  # broadcast over K
        # [channels, K, S] -> [B, channels, K, S]
        return pe.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)



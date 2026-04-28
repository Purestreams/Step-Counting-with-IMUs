from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.drop(h)
        return F.gelu(h + x)


class CausalTCNStepNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        channels: int = 64,
        kernel_size: int = 5,
        num_blocks: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        blocks = []
        for i in range(num_blocks):
            blocks.append(CausalResBlock(channels, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.event_head = nn.Conv1d(channels, 1, kernel_size=1)
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        h = self.tcn(h)
        event_logits = self.event_head(h).squeeze(1)
        count_pred = self.count_head(h).squeeze(-1)
        return event_logits, count_pred


class LSTMStepNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.event_head = nn.Linear(hidden_size, 1)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        h, _ = self.lstm(x)
        event_logits = self.event_head(h).squeeze(-1)
        pooled = h.mean(dim=1)
        count_pred = self.count_head(pooled).squeeze(-1)
        return event_logits, count_pred


def build_model(model_name: str):
    name = model_name.lower()
    if name == "tcn":
        return CausalTCNStepNet()
    if name == "lstm":
        return LSTMStepNet()
    raise ValueError(f"Unsupported model: {model_name}")

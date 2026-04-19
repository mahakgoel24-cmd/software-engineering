import torch
import torch.nn as nn
import torch.nn.functional as F

class FairRankingModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        raw_score = self.net(x).squeeze(1)
        return torch.sigmoid(raw_score)  # ⬅️ normalized to [0, 1]

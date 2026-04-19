import torch
import torch.nn as nn
import torch.nn.functional as F

class SkillInferenceModel(nn.Module):
    def __init__(self, embedding_dim, meta_dim, output_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + meta_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(128, output_dim)
        self.var_head = nn.Linear(128, output_dim)

    def forward(self, emb, meta):
        x = torch.cat([emb, meta], dim=1)
        h = self.fc(x)

        # Normalize latent representation for stability
        h = F.normalize(h, p=2, dim=1)

        mean = self.mean_head(h)
        log_var = self.var_head(h)

        # Clamp variance to avoid explosion
        log_var = torch.clamp(log_var, min=-5.0, max=2.0)

        return mean, log_var

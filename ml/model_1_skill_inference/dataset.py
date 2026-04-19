import torch
from torch.utils.data import Dataset

class SubmissionDataset(Dataset):
    def __init__(self, embeddings, metadata, targets=None):
        self.embeddings = embeddings
        self.metadata = torch.tensor(metadata, dtype=torch.float32)

        if targets is not None:
            targets = torch.tensor(targets, dtype=torch.float32)
            self.targets = torch.clamp(targets, 0.0, 1.0)
        else:
            self.targets = None

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x_text = self.embeddings[idx]
        x_meta = self.metadata[idx]

        if self.targets is not None:
            y = self.targets[idx]
            return x_text, x_meta, y

        return x_text, x_meta

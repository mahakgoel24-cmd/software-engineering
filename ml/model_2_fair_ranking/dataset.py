import torch
from torch.utils.data import Dataset

class RankingDataset(Dataset):
    def __init__(self, features, user_ids):
        """
        features: List[List[float]] -> shape [N, D]
        user_ids: List[str] (Supabase UUIDs)
        """
        self.X = torch.tensor(features, dtype=torch.float32)
        self.user_ids = list(user_ids)

        assert len(self.X) == len(self.user_ids), "Features and user_ids mismatch"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.user_ids[idx]

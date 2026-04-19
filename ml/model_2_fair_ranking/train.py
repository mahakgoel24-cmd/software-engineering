import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import RankingDataset
from .model import FairRankingModel
from .fairness import fairness_regularization

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------
# TRAINING DATA FORMAT (REALISTIC)
# ------------------------------------------------
# This would normally come from DB + Model 1 logs
training_features = [
    # clarity, depth, creativity, quality, uncertainty,
    # semantic_match, experience, rating, completion
    [0.72, 0.68, 0.64, 0.69, 0.02, 0.91, 3, 4.6, 0.92],
    [0.75, 0.71, 0.59, 0.70, 0.01, 0.74, 1, 4.1, 0.88],
    [0.66, 0.63, 0.70, 0.68, 0.03, 0.62, 2, 3.9, 0.80]
]

user_ids = [
    "uuid-user-1",
    "uuid-user-2",
    "uuid-user-3"
]

dataset = RankingDataset(training_features, user_ids)
loader = DataLoader(dataset, batch_size=3, shuffle=True)

model = FairRankingModel(input_dim=len(training_features[0])).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------
for epoch in range(50):
    model.train()

    for X, _ in loader:
        X = X.to(DEVICE)

        scores = model(X)
        uncertainty = X[:, 4]  # uncertainty_mean

        ranking_loss = -torch.mean(scores)
        fairness_loss = fairness_regularization(scores, uncertainty)

        loss = ranking_loss + fairness_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

# ------------------------------------------------
# SAVE MODEL
# ------------------------------------------------
import os
os.makedirs("ml/artifacts/model_2", exist_ok=True)
torch.save(model.state_dict(), "ml/artifacts/model_2/fair_ranking.pt")

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import os

from ml.common.embeddings import TextEmbedder
from .dataset import SubmissionDataset
from .model import SkillInferenceModel
from .loss import gaussian_nll_loss
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using device:", DEVICE)

# ------------------------------
# Dummy data (temporary)
# ------------------------------
from .synthetic_data import generate_dataset

texts, metadata, targets = generate_dataset(n=500)

# ------------------------------
# Embeddings
# ------------------------------
embedder = TextEmbedder()
embeddings = embedder.encode(texts)

# ------------------------------
# Dataset & split
# ------------------------------
dataset = SubmissionDataset(embeddings, metadata, targets)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2)

# ------------------------------
# Model
# ------------------------------
model = SkillInferenceModel(
    embedding_dim=embeddings.shape[1],
    meta_dim=len(metadata[0]),
    output_dim=4
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------
# Training loop with early stopping
# ------------------------------
best_val_loss = float("inf")
patience = 0
max_patience = 3

for epoch in range(50):
    model.train()
    train_loss = 0.0

    for emb, meta, y in train_loader:
        emb = emb.to(DEVICE)
        meta = meta.to(DEVICE)
        y = y.to(DEVICE)

        mean, log_var = model(emb, meta)
        loss = gaussian_nll_loss(mean, log_var, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for emb, meta, y in val_loader:
            emb = emb.to(DEVICE)
            meta = meta.to(DEVICE)
            y = y.to(DEVICE)

            mean, log_var = model(emb, meta)
            loss = gaussian_nll_loss(mean, log_var, y)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # ---- Early stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
    else:
        patience += 1
        if patience >= max_patience:
            print("Early stopping triggered.")
            break

# ------------------------------
# Save trained model
# ------------------------------
os.makedirs("ml/artifacts/model_1", exist_ok=True)

MODEL_PATH = "ml/artifacts/model_1/skill_inference.pt"

torch.save(model.state_dict(), MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
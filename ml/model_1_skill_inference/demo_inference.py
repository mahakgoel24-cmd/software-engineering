import torch
from ml.common.embeddings import TextEmbedder
from .model import SkillInferenceModel
from .inference import infer

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Load trained model
model = SkillInferenceModel(384, 3, 4).to(DEVICE)
model.load_state_dict(
    torch.load("ml/artifacts/model_1/skill_inference.pt", map_location=DEVICE)
)

# Simulate a real developer submission
submission_text = [
    "I designed a scalable backend using FastAPI with clear separation of concerns."
]

metadata = torch.tensor([[5.5, 2, 2]], dtype=torch.float32)

embedder = TextEmbedder()
emb = embedder.encode(submission_text).to(DEVICE)
meta = metadata.to(DEVICE)

result = infer(model, emb, meta)

print("Model 1 Output Scores:")
for k, v in result.items():
    print(f"{k}: {v}")

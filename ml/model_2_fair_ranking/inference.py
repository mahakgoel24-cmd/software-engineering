import torch
from .model import FairRankingModel

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

def rank_candidates(features, user_ids, model_path):
    """
    features: List[List[float]]  -> shape (N, 9)
    user_ids: List[str]
    returns: List[{user_id, score}]
    """

    X = torch.tensor(features, dtype=torch.float32).to(DEVICE)

    model = FairRankingModel(input_dim=X.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        scores = model(X).cpu().tolist()

    # ✅ THIS IS THE FIX
    ranked = [
        {
            "user_id": user_ids[i],
            "score": float(scores[i])
        }
        for i in range(len(user_ids))
    ]

    return ranked

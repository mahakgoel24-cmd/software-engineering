import torch

from ml.model_2_fair_ranking.inference import rank_candidates

# --------------------------------------------------
# SIMULATED REALISTIC INPUT (AS IF FROM BACKEND)
# --------------------------------------------------

# Supabase-style UUIDs (strings, NOT ints)
user_ids = [
    "uid_7c91",
    "uid_a23f",
    "uid_f991",
    "uid_81be"
]

# Features per candidate in SAME ORDER as user_ids
# [
#   clarity,
#   technical_depth,
#   creativity,
#   overall_quality,
#   uncertainty_mean,
#   semantic_match_score,
#   years_experience,
#   avg_rating,
#   completion_rate
# ]
features = [
    [0.72, 0.68, 0.64, 0.69, 0.02, 0.91, 3, 4.6, 0.92],  # strong match
    [0.78, 0.75, 0.60, 0.72, 0.01, 0.70, 1, 4.1, 0.88],  # good skills, lower match
    [0.65, 0.62, 0.70, 0.66, 0.03, 0.60, 2, 3.9, 0.80],  # creative, weaker fit
    [0.60, 0.58, 0.55, 0.59, 0.04, 0.40, 4, 4.8, 0.95],  # high rating, poor match
]

MODEL_PATH = "ml/artifacts/model_2/fair_ranking.pt"

# --------------------------------------------------
# RUN RANKING
# --------------------------------------------------

ranked_user_ids, ranked_scores = rank_candidates(
    features=features,
    user_ids=user_ids,
    model_path=MODEL_PATH
)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n=== MODEL 2 RANKING RESULTS ===\n")

for i, (uid, score) in enumerate(zip(ranked_user_ids, ranked_scores), start=1):
    print(f"Rank {i}: {uid} | Score: {score:.4f}")

print("\nTop candidate UID:", ranked_user_ids[0])

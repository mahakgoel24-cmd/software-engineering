# ml/model_3_qml_optimizer/test_quantum.py

from ml.model_3_qml_optimizer.inference import quantum_select_top_10_percent

# ---------------------------
# Simulated output from Model 2
# ---------------------------
ranked_user_ids = [
    "uid_1a",
    "uid_2b",
    "uid_3c",
    "uid_4d",
    "uid_5e",
    "uid_6f",
    "uid_7g",
    "uid_8h",
    "uid_9i",
    "uid_10j"
]

scores = [
    0.95, 0.93, 0.92, 0.91, 0.90,
    0.89, 0.88, 0.87, 0.86, 0.85
]

# ---------------------------
# Business constraints
# ---------------------------
salaries = [60, 55, 50, 65, 58, 62, 48, 52, 57, 54]  # hourly cost
hours =    [30, 28, 35, 40, 25, 32, 20, 22, 30, 26]
availability = [40, 30, 40, 35, 30, 40, 25, 25, 35, 30]

# ---------------------------
# Run quantum selection
# ---------------------------
selected = quantum_select_top_10_percent(
    ranked_user_ids=ranked_user_ids,
    scores=scores,
    salaries=salaries,
    hours=hours,
    availability=availability
)

print("\n=== QUANTUM SELECTION RESULT ===")
print("Selected candidate IDs:")
for uid in selected:
    print(uid)

# ml/model_3_qml_optimizer/postprocess.py

def extract_selected_users(solution, user_ids, k):
    selected = [
        user_ids[i]
        for i, bit in enumerate(solution)
        if bit == 1
    ]

    return selected[:k]

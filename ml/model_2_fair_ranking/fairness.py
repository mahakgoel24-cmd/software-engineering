import torch

def fairness_regularization(scores, uncertainty, alpha=0.2):
    """
    Penalize extreme domination while respecting uncertainty
    """
    score_variance = torch.var(scores)
    uncertainty_penalty = torch.mean(uncertainty)

    return alpha * score_variance + (1 - alpha) * uncertainty_penalty

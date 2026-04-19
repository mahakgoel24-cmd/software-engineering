import torch

def gaussian_nll_loss(mean, log_var, target):
    var = torch.exp(log_var)
    loss = (mean - target) ** 2 / (var + 1e-6) + log_var
    return torch.mean(loss)

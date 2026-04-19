import torch

def infer(model, emb, meta):
    device = next(model.parameters()).device

    emb = emb.to(device)
    meta = meta.to(device)

    model.eval()
    with torch.no_grad():
        mean, log_var = model(emb, meta)
        uncertainty = torch.exp(log_var)

    return {
        "clarity": float(mean[0, 0].cpu()),
        "technical_depth": float(mean[0, 1].cpu()),
        "creativity": float(mean[0, 2].cpu()),
        "overall_quality": float(mean[0, 3].cpu()),
        "uncertainty": uncertainty[0].cpu().tolist()
    }

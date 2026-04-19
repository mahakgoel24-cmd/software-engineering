import torch
import torch.nn.functional as F

def compute_semantic_match(
    candidate_embeddings: torch.Tensor,
    project_embedding: torch.Tensor
) -> torch.Tensor:
    """
    candidate_embeddings: Tensor [N, D]
    project_embedding:   Tensor [1, D]

    returns: Tensor [N] with cosine similarity scores in [0, 1]
    """

    if project_embedding.dim() == 1:
        project_embedding = project_embedding.unsqueeze(0)

    # Expand project embedding to match batch size
    project_embedding = project_embedding.expand(
        candidate_embeddings.size(0), -1
    )

    similarity = F.cosine_similarity(
        candidate_embeddings,
        project_embedding,
        dim=1
    )

    # Clamp for numerical safety
    similarity = torch.clamp(similarity, 0.0, 1.0)

    return similarity

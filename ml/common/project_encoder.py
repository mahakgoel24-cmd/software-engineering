import torch
from ml.common.embeddings import TextEmbedder

class ProjectEncoder:
    def __init__(self):
        self.embedder = TextEmbedder()

    def encode_project(self, requirements_text: str) -> torch.Tensor:
        """
        requirements_text: combined string of
        - project needs
        - tech stack
        - timelines
        - expectations

        returns: torch.Tensor [1, D]
        """
        embedding = self.embedder.encode([requirements_text])
        return embedding

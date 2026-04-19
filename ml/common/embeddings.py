from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self):
        # 🔒 MUST MATCH TRAINING
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

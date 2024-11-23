from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction, Embeddings, Documents

class CustomSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path):
        # device="cuda" - To use this we need pytorch with cuda
        self.model = SentenceTransformer(model_path)

    def __call__(self, input: Documents) -> Embeddings:
        # Generate embeddings using the local model
        embeddings = self.model.encode(input)
        return embeddings
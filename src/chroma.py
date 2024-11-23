import os
import chromadb
from typing import List
from src.embedding_func import CustomSentenceTransformerEmbeddingFunction

class ChromaDatabase:
    def __init__(self, host: str, port: int, collectionName: str):
        self.client = chromadb.HttpClient(host=host, port=port)

        self.custom_ef = CustomSentenceTransformerEmbeddingFunction( self.GetModelDirectory() )
        self.collection = self.client.get_or_create_collection(name=collectionName, embedding_function=self.custom_ef)


    def QueryPrompt(self, prompt: str, neighbours: int, tags: List[str]):
        results = self.collection.query(query_texts=[prompt],
                                        n_results=neighbours,
                                        where={"tag": {"$in": [tag for tag in tags]}})
        self.DisplayNeighbours(results)
        return results


    def DisplayNeighbours(self, neighbours: chromadb.QueryResult):
        for i, doc in enumerate(neighbours['documents'][0]):
            print(f"Document {i}: {doc}")

    
    def GetModelDirectory(self):
        current_path = os.path.abspath(__file__)
        current_path = os.path.dirname(current_path)
        model_directory = os.path.join(current_path, '../models/all-mpnet-base-v2/')
        return model_directory
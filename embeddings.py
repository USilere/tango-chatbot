import os
from typing import Any
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions 

class Embeddings:
    def __init__(self):
        self.client = chromadb.Client()
        self.embedding_function_model_name = "all-MiniLM-L6-v2"
        self.sbert_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_function_model_name
        )
        self.vector_db = None
    
    def embed_data_chunks(self, chunks):
        self.vector_db = self.client.get_or_create_collection(
            name="my_collection",
            embedding_function=self.sbert_ef,
        )
        for i, chunk in enumerate(chunks):
            self.vector_db.add(
                documents=[chunk.page_content],
                ids=[f"chunk_{i}"]
            )

    # def retrieve_data(self, user_query: str) -> str :
    #     print(f"DEBUG: Incoming user query for retrieval: {user_query}")
    #     retrieved_data: str = ""
    #     docs = self.vector_db.query(
    #         query_texts=[user_query],
    #         n_results=3
    #     )
    #     for matches in docs["documents"][0]:
    #         retrieved_data.join(matches)
    
    #     print(f">>>>> Retrieved string based on query: {retrieved_data}")

    #     if len(retrieved_data) == 0:
    #         retrieved_data = "No data matching query found"
        
    #     return retrieved_data
    
    def retrieve_data(self, query: str) -> None:
        """
        Retrieves and prints the top matching documents from the collection based on the provided query.
        Args:
            query (str): The search query used to find relevant documents.
        Returns:
            None
        """
        try:
            collection: Any = self.client.get_or_create_collection(
                name="my_collection",
                embedding_function=self.sbert_ef
            )
            self.collection: Any = collection
            docs: dict = self.collection.query(
                query_texts=[query],
                n_results=3
            )
            for matches in docs["documents"][0]:
                print(matches)
        except Exception as err:
            print(f"Failed to retrieve data: {err}")
        return matches
    
 

from typing import Any
from langchain_core.documents import Document
from typing_extensions import List
from langgraph.graph import StateGraph, START
from langchain import hub
from langchain.chat_models import init_chat_model
import os

class StateAgent():
    def __init__(self):
        self.user_query: str
        self.context: List[Document]
        self.answer: str

        self.prompt = hub.pull("rlm/rag-prompt")

        self.llm = None
        self.init_llm()

    def init_llm(self) -> Any:
        if self.verify_api_key_supplied():
            self.llm = init_chat_model(
                "gemini-2.5-flash", 
                model_provider="google_genai",
            )
            return
        # Clean up
        print("Configure the Gemini API key")
    
    def verify_api_key_supplied(self):
        # TODO: This is all temp. Clean up
        # Consider support for different LLM's
        if not 'GOOGLE_API_KEY' in os.environ:
            print("Gemini API key not supplied. TEMP: Applying")
            # TODO: Get key if not yet supplied
            # TODO: Remove - personal key, do not publish
            os.environ['GOOGLE_API_KEY'] = "Placeholder_key"
            return True
        return True
    
    def response_generation(self, ret_documents: Any):
        # Create a prompt template to that tells the LLM how to
        # distinguish a code block from the source PDF
        ip_prompt = self.prompt.invoke(
            {"question": self.user_query, 
             "context": ret_documents}
        )

        # Validate that an API key has been provided
        if self.verify_api_key_supplied():
            op_response = self.llm.invoke(ip_prompt)
            self.answer = op_response
            # For now print it to the command line
            print(f"Got user query: {self.user_query}\n\n")
            print(f"LLM Response: {self.answer}")

    def receive_user_query(self, incoming_query: str) -> Any:
        self.user_query = incoming_query
        # TODO: Add consideration of chat history, reformulate user
        # query with chat history consideration, and assign back
        # to the 

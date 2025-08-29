from typing import Any
from langchain_core.documents import Document
from typing_extensions import List
from langgraph.graph import StateGraph, START
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
import os
from langchain.prompts import PromptTemplate

from models import prompt_templates

class StateAgent():
    def __init__(self):
        self.user_query: str
        self.context: List[Document]
        self.answer: str

        self.verify_api_key_supplied()

        self.prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=prompt_templates.rag_agent_response_prompt
        )
        self.llm = None
        self.init_llm()

        self.current_chat_history = InMemoryChatMessageHistory()

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
            api_key = input("Please enter your Gemini API key: ")
            os.environ['GOOGLE_API_KEY'] = api_key
        return True
    
    async def response_generation(self, ret_documents: Any):
        # Create a prompt template to that tells the LLM how to
        # distinguish a code block from the source PDF
        ip_prompt = self.prompt.invoke(
            {"question": self.user_query, 
             "context": ret_documents}
        )
        # Validate that an API key has been provided

        if self.verify_api_key_supplied():
            op_response = self.llm.invoke(ip_prompt)
            self.answer = op_response.content
            print(f"LLM Response: {self.answer}")
        
        # Update history
        self.current_chat_history.add_ai_message([
            self.answer,
        ])

        # Debug: Print the current message history
        debug_hist = await self.current_chat_history.aget_messages()
        # Trim history to last 10 entries if needed
        # The API doesnt allow clearing a certain number of
        # messages so hold the history, trim it, then rewrite
        if len(debug_hist) > 10:
            debug_hist = debug_hist[-10:]
            self.current_chat_history.add_messages(debug_hist)


    async def receive_user_query(self, incoming_query: str) -> Any:
        self.user_query = incoming_query
        self.current_chat_history.add_user_message(self.user_query)
        # Reformulate the current query to incorporate the message history
        history = await self.current_chat_history.aget_messages()

        prompt_temp = PromptTemplate(
            input_variables=["chat_history", "user_input"],
            template=prompt_templates.question_reformulation_prompt
        )
        formatted_query = prompt_temp.format(
            chat_history=history,
            user_input=incoming_query
        )

        if self.verify_api_key_supplied():
            reformulated_query = self.llm.invoke(formatted_query)
            self.user_query = reformulated_query
            print(f"DEBUG: Reforumlated query: {self.user_query}")

    
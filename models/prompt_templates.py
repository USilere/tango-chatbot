
rag_agent_response_prompt = """
    You are an assistant for question-answering tasks. Use the \
    following pieces of retrieved context to answer the question. \
    The context may include explanation paragraphs and/or code snippets.\
    You may answer the query in up to 10 sentences, but try to use less \
    where possible. Where the context provides relevant code snippets \
    include them at the end of the response. \
    If you don't know the answer, just say that you don't know. \
    
    Question: {question} 
    Context: {context}
    Answer:
    """

question_reformulation_prompt = """
    Given a conversation history and the latest user question \
    which might reference context in the chat history, formulate \
    a standalone question which can be understood without the \
    chat history. Do NOT answer the question, just reformulate it
    if needed and otherwise return it as is.
    
    Conversation history:
    {chat_history}

    Current user question:
    {user_input}
    
    Refined Query:
    """
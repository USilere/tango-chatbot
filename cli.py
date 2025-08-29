from embeddings import Embeddings
from source_data_chunker import SourceDataProcessor
from models.enums import SeparatorSelection
from stateful_agent import StateAgent

import glob

import asyncio


async def main():
    source_pdf_path = "./source_data/"
    source_files = glob.glob(source_pdf_path + '/*.pdf', recursive=True)
    
    document_processor = SourceDataProcessor
    embedding_model = Embeddings()
    
    for current_file in source_files:
        print(f">>> Processing source file: {current_file}")
        document = document_processor.source_data_loader(source_pdf_path)

        # 1: Create data chunks
        data_chunks = None
        if document is not None:
            print(f"Chunking source data from file [{current_file}]...")
            data_chunks = document_processor.source_data_chunker(
                document, 
                SeparatorSelection.DOUBLE_NEW_LINE
            )
            print(f"Data chunked successfully")
        else:
            print("PDF loading/chunking failed.")
            continue

        # 2: Pass data chunks to embedding model to store in DB
        if data_chunks is not None:
            print(f"Applying embedding model to [{current_file}] chunks...")
            embedding_model.embed_data_chunks(data_chunks)
            print("Embedding model applied successfully")
        else:
            print("Could not apply embedding model due to empty chunk list")
            

    # 3: Initialise the Stateful agent to enable LLM integration
    print("Initializing LLM agent...")
    llm_agent = StateAgent()
    
    # 4: get user prompt;
    #    4.1: Fetch matches from vector DB
    #    4.2: Pass matches to LLM and print response
    print("Enter your prompt (Ctrl+C to exit):")
    try:
        while True:
            user_input = input("> ")
            await llm_agent.receive_user_query(user_input)
            retrieved_documents = embedding_model.retrieve_data(user_input)
            response = await llm_agent.response_generation(retrieved_documents)
            print(response)
    except KeyboardInterrupt:
        print("\nExiting")

if __name__ == "__main__":
    print("Initializing...")
    asyncio.run(main())
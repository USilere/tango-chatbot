import os
import argparse
from embeddings import Embeddings
from source_data_chunker import SourceDataProcessor
from models.enums import SeparatorSelection
from stateful_agent import StateAgent


def main():
    # source_pdf_path = "./source_data/tango_91.pdf"
    source_pdf_path = "./source_data/Example_code_snippets_pdf.pdf"
    if not os.path.isfile(source_pdf_path):
        print(f"File not found: {source_pdf_path}")
        return
    
    document_processor = SourceDataProcessor
    document = document_processor.source_data_loader(source_pdf_path)

    # 1: Create data chunks
    data_chunks = None
    if document is not None:
        print("Chunking source data...")
        # data_chunks = document_processor.source_data_chunker(document, SeparatorSelection.SINGLE_NEW_LINE)
        data_chunks = document_processor.source_data_chunker(document, SeparatorSelection.TRIPLE_HASH)
        print(f"Data chunked successfully")
    else:
        print("PDF loading/chunking failed.")
        return

    # 2: Pass data chunks to embedding model to store in DB
    if data_chunks is not None:
        print("Applying embedding model...")
        embedding_model = Embeddings()
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
            # Clean this up
            llm_agent.receive_user_query(user_input)
            retrieved_documents = embedding_model.retrieve_data(user_input)
            print(f"DEBUG: Returned documents based on query: {retrieved_documents}")
            response = llm_agent.response_generation(retrieved_documents)
            print(response)
    except KeyboardInterrupt:
        print("\nExiting")


if __name__ == "__main__":
    print("Initializing...")
    main()

    # # 1. Create a parser object
    # parser = argparse.ArgumentParser(
    #     description="A RAG chatbot that answers questions from a PDF document."
    # )

    # # 2. Add a command-line argument for the query
    # parser.add_argument(
    #     "query",
    #     type=str,
    #     help="The question you want to ask the chatbot."
    # )
    # parser.add_argument(
    #     "--pdf_path",
    #     type=str,
    #     default="PyTango News.pdf",
    #     help="Path to the PDF document to be used as the knowledge base."
    # )
    # # 3. Parse the arguments from the command line
    # args = parser.parse_args()
    # query = args.query
    # pdf_path = args.pdf_path
    # # Initialize the data processor
    # data_processor = SourceDataProcessor()

    # # Initialize the embeddings manager
    # embeddings_manager = Embeddings()

    # ------
    

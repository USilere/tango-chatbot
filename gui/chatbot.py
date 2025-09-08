import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from embeddings import Embeddings
from source_data_chunker import SourceDataProcessor
from models.enums import SeparatorSelection
from stateful_agent import StateAgent


import glob
import asyncio

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
 


llm_agent = StateAgent()


# Theme colors inspired by TANGO logo
BG_COLOR = "#1E1E1E"   # dark background
FG_COLOR = "#E0E0E0"   # light grey text
ACCENT_GREEN = "#7AC943"  # Tango green
ACCENT_GREY = "#666666"   # Tango grey

# Setup window
root = tk.Tk()
root.title("TANGO Chatbot")
tango_icon = tk.PhotoImage(file="/tango-chatbot/tango-chatbot/gui/icons/Tango Controls Icon.png")
# root.wm_iconphoto(True, tango_icon)
root.geometry("700x650")
root.configure(bg=BG_COLOR)

# Header
header = tk.Label(
    root, 
    text="TANGO Chatbot",
    font=("Segoe UI", 16, "bold"),
    image=tango_icon, 
    compound=tk.LEFT, 
    bg=BG_COLOR, 
    fg=ACCENT_GREEN,
    pady=10
)
header.pack()

# Matches log area
matches_log = scrolledtext.ScrolledText(
    root, bd=1, bg=ACCENT_GREY, fg="white",
    font=("Segoe UI", 18), wrap=tk.WORD, height=0.2
)

query_label = tk.Label(root, text="Enter your query:", font=("Segoe UI", 18),
                       bg=BG_COLOR, fg=ACCENT_GREEN)

query_entry = tk.Entry(root, bd=1, bg="black", fg="white",
                       font=("Segoe UI", 12), width=65, insertbackground="white")

answer_label = tk.Label(root, text="LLM Response:", font=("Segoe UI", 18),
                        bg=BG_COLOR, fg=ACCENT_GREEN)

answer_textbox = scrolledtext.ScrolledText(
    root, bd=1, bg=ACCENT_GREY, fg="white",
    font=("Segoe UI", 12), wrap=tk.WORD, height=10
)

def on_submit():
    query = query_entry.get().strip()
    if not query:
        return
    async def run_agent():
        # Step 1: send query to agent
        await llm_agent.receive_user_query(query)

        ret_documents = embedding_model.retrieve_data(query)
        await llm_agent.response_generation(ret_documents)

        matches_log.insert(tk.END, " References:\n")
        for c in ret_documents:
            matches_log.insert(tk.END, f" - {c}\n")

        # Step 2: generate LLM response
        await llm_agent.response_generation(ret_documents)

        # Step 3: show answer
        answer_textbox.delete("1.0", tk.END)
        answer_textbox.insert(tk.END, llm_agent.answer)

        # Clear input
        query_entry.delete(0, tk.END)
    asyncio.run(run_agent())
submit_button = tk.Button(root, text="Submit",
                          font=("Segoe UI", 12, "bold"),
                          bg=ACCENT_GREEN, fg="black",
                          command=on_submit, relief="flat", padx=15, pady=5)

# Layout
matches_log.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
query_label.pack(pady=5)
query_entry.pack(pady=5)
submit_button.pack(pady=10)
answer_label.pack(pady=5)
answer_textbox.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

matches_log.insert(tk.END, "Welcome to the Tango Knowledge Chatbot!\n")

root.mainloop()

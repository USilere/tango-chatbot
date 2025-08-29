from typing import Any, Optional
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.enums import SeparatorSelection

import os
from pathlib import Path

class SourceDataProcessor:
    def source_data_loader(source_data_path: str) -> Any:
        """
        Loads a PDF document from the specified file path.

        Args:
            source_data_path (str): The path to the source PDF file.

        Returns:
            Any: The loaded PDF document object if successful, otherwise None.

        Raises:
            Prints an error message if the PDF file cannot be loaded.
        """
        src_doc = None
        try:
            loader = GenericLoader(
                blob_loader=FileSystemBlobLoader(
                    path=source_data_path,
                    glob="*.pdf",
                ),
                blob_parser=PyPDFParser(),
            )
            src_doc = loader.load()
        except Exception as err:
            print(f"Failed to open source PDF file: {err}")
        return src_doc


    def source_data_chunker(source_document: Any, rec_chuncker_separtor: Optional[SeparatorSelection] = None,) -> Any:
        """
        Splits the given source document into chunks based on the specified separator configuration.

        Args:
            rec_chuncker_separtor (SeparatorSelection): Enum value specifying the type of separator to use for chunking.
            source_document (Any): The source document to be split into chunks.

        Returns:
            Any: A list of text chunks obtained by splitting the source document.

        Notes:
            - Uses RecursiveCharacterTextSplitter to perform the splitting.
            - The chunk size is set to 1500 characters with an overlap of 200 characters.
            - Separator configuration is determined by the rec_chuncker_separtor argument.
        """
        separator_config = []
        split_data = None
        if rec_chuncker_separtor == SeparatorSelection.SINGLE_NEW_LINE:
            separator_config = ["\n"]
        elif rec_chuncker_separtor == SeparatorSelection.DOUBLE_NEW_LINE:
            separator_config = ["\n\n", "\n"]
        elif rec_chuncker_separtor == SeparatorSelection.DOUBLE_NEW_LINE:
            separator_config = ["###"]
        else:
            separator_config = ["\n\n", "\n", " ", ""]

        splitter = RecursiveCharacterTextSplitter(
            separators=separator_config,
            chunk_size=1500,
            chunk_overlap=200,
        )
        try:
            split_data = splitter.split_documents(source_document)
        except Exception as err:
            print(f"Failed to split source data: {err}")
        return split_data

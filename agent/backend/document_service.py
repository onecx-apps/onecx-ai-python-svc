import os
import zipfile
import copy
import requests
from omegaconf import DictConfig
from agent.utils.configuration import load_config
from agent.utils.utility import replace_multiple_whitespaces
from agent.utils.utility import extract_procedures_from_issue, get_issueid_score_dict
from loguru import logger
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from typing import List, Optional, Tuple
from langchain.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
from agent.backend.qdrant_service import get_db_connection
from qdrant_client.http import models as qdrant_models

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
ACTIVATE_RERANKER = os.getenv('ACTIVATE_RERANKER')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_PORT = os.getenv('QDRANT_PORT')
SCORE_THREASHOLD = os.getenv('SCORE_THREASHOLD', .7)
IMAGES_LOCATION = os.getenv('IMAGES_LOCATION')



class DocumentService():
    def __init__(self):
        self.embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = get_db_connection(embedding_model=self.embedding_model)
        
    def embed_directory(self, dir: str) -> None:
        """Embeds the documents in the given directory in the llama2 database.

        This method uses the Directory Loader for PDFs and the PyPDFLoader to load the documents.
        The documents are then added to the Qdrant DB which embeds them without deleting the old collection.

        Args:
            dir (str): The directory containing the PDFs to embed.

        Returns:
            None
        """
        logger.info(f"Logged directory:  {dir}")
        loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
        length_function = len
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "; ", "! ", "? ", "# "],
            chunk_size=500,
            chunk_overlap=50,
            length_function=length_function,
        )
        docs = loader.load_and_split(splitter)
        logger.info(f"Loaded {len(docs)} documents.")
        text_list = [doc.page_content for doc in docs]
        metadata_list = [doc.metadata for doc in docs]
        self.vector_store.add_texts(texts=text_list, metadatas=metadata_list)
        logger.info("SUCCESS: Texts embedded.")

    def embed_zip(self, filename: str, dir: str) -> None:
        """
        Args:
            filename (str): name of the zip file
            dir (str): The directory containing the extracted files to embed.

        Returns:
            None
        """
        logger.info(f"Logged extraction directory:  {dir}")

        # Extract the contents of the zip file
        zip_file_path = os.path.join(dir, filename)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dir)

        loader = DirectoryLoader(dir, glob="**/*.json",  loader_cls=TextLoader)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents.")
        trans_docs = extract_procedures_from_issue(docs, IMAGES_LOCATION)
        self.vector_store.add_documents(trans_docs)
        logger.info(f"SUCCESS: {len(trans_docs)} Texts embedded.")

    def embed_text(self, text: str, file_name: str, seperator: str) -> None:
        """Embeds the given text in the llama2 database.

        Args:
            text (str): The text to be embedded.


        Returns:
            None
        """
        # split the text at the seperator
        text_list: List = text.split(seperator)

        # check if first and last element are empty
        if not text_list[0]:
            text_list.pop(0)
        if not text_list[-1]:
            text_list.pop(-1)

        metadata = file_name
        # add _ and an incrementing number to the metadata
        metadata_list: List = [{"source": f"{metadata}_{str(i)}", "page": 0} for i in range(len(text_list))]

        self.vector_store.add_texts(texts=text_list, metadatas=metadata_list)
        logger.info("SUCCESS: Text embedded.")


    def embed_folder(self, folder: str, seperator: str) -> None:
        """Embeds text files in the llama2 database.

        Args:
            folder (str): The folder containing the text files to embed.
            seperator (str): The seperator to use when splitting the text into chunks.

        Returns:
            None
        """
        # iterate over the files in the folder
        for file in os.listdir(folder):
            # check if the file is a .txt or .md file
            if not file.endswith((".txt", ".md")):
                continue

            # read the text from the file
            with open(os.path.join(folder, file)) as f:
                text = f.read()

            text_list: List = text.split(seperator)

            # check if first and last element are empty
            if not text_list[0]:
                text_list.pop(0)
            if not text_list[-1]:
                text_list.pop(-1)

            # ensure that the text is not empty
            if not text_list:
                raise ValueError("Text is empty.")

            logger.info(f"Loaded {len(text_list)} documents.")
            # get the name of the file
            metadata = os.path.splitext(file)[0]
            # add _ and an incrementing number to the metadata
            metadata_list: List = [{"source": f"{metadata}_{str(i)}", "page": 0} for i in range(len(text_list))]
            self.vector_store.add_texts(texts=text_list, metadatas=metadata_list)

        logger.info("SUCCESS: Text embedded.")


    def search_documents(self, query: str, amount: int) -> List[Tuple[Document, float]]:
        """Searches the documents in the Qdrant DB with a specific query.

        Args:
            query (str): The question for which documents should be searched.

        Returns:
            List[Tuple[Document, float]]: A list of search results, where each result is a tuple
            containing a Document object and a float score.
        """
        docs = self.vector_store.similarity_search_with_score(query, k=amount, score_threshold=SCORE_THREASHOLD)
        logger.debug(f"\nNumber of documents: {len(docs)}")

        if docs is not None and len(docs) > 0:
            logger.debug("SUCCESS: Documents found after similarity_search_with_score.")

            # Extract issueIds with its highes score
            score_issueIds = get_issueid_score_dict(docs)


            for element in docs:
                document, score = element
                logger.debug(f"\n Document found with score: {score}")
                logger.debug(replace_multiple_whitespaces(document.page_content))
                logger.debug(document.metadata)

            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                "filter": qdrant_models.Filter(                    
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.type",
                            match=qdrant_models.MatchValue(value="issue")
                        ),
                        qdrant_models.FieldCondition(
                            key="metadata.issueId",
                            match=qdrant_models.MatchAny(any=list(score_issueIds.keys()))
                        )                        
                    ]
                ),
                "k": 3})


            filtered_docs = retriever.get_relevant_documents(query)
            logger.debug(f"\n {len(filtered_docs)} filtered documents found")
            logger.info(f"Filtered docs {filtered_docs} ")

            if ACTIVATE_RERANKER == "True":
                embedding = self.embedding_model
                
                retriever = self.vector_store.from_documents(filtered_docs, embedding, api_key=QDRANT_API_KEY, url=QDRANT_URL, collection_name="temp_ollama").as_retriever()

                rerank_compressor = CohereRerank(user_agent="my-app", model="rerank-multilingual-v2.0", top_n=3)
                splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", "; ", "! ", "? ", "# "],chunk_size=120, chunk_overlap=20)
                redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
                relevant_filter = EmbeddingsFilter(embeddings=embedding)
                pipeline_compressor = DocumentCompressorPipeline(
                    transformers=[splitter, redundant_filter, relevant_filter, rerank_compressor]
                )
                compression_retriever1 = ContextualCompressionRetriever(base_compressor=rerank_compressor, base_retriever=retriever)
                compressed_docs = compression_retriever1.get_relevant_documents(query)

                for docu in compressed_docs:
                    logger.info(f"Context after reranking: {replace_multiple_whitespaces(docu.page_content)}")

                #Delete the temporary qdrant collection which is used for the base retriever
                url = f"{QDRANT_URL}:{QDRANT_PORT}/collections/temp_ollama"
                headers = {"Content-Type": "application/json", "api-key": QDRANT_API_KEY}
                requests.delete(url, headers=headers)

                return compressed_docs
            else:
                #Logic for none-reranking needs to be implemented here
                #filtered_docs = [t[0] for t in docs]
                return filtered_docs
        return None

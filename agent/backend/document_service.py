import os
import zipfile
import copy
import requests
from flashrank import Ranker
from omegaconf import DictConfig
from agent.utils.configuration import load_config
from agent.utils.utility import replace_multiple_whitespaces
from agent.utils.utility import extract_procedures_from_issue, get_issueid_score_dict, get_issueid_dict
from loguru import logger
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers.document_compressors import FlashrankRerank

from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from typing import List, Optional, Tuple
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
from agent.backend.qdrant_service import get_db_connection
from qdrant_client.http import models as qdrant_models
from typing import TYPE_CHECKING, Dict, Optional, Sequence
from langchain.callbacks.manager import Callbacks
from flashrank import Ranker, RerankRequest

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
ACTIVATE_RERANKER = os.getenv('ACTIVATE_RERANKER')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_PORT = os.getenv('QDRANT_PORT')
SCORE_THREASHOLD = os.getenv('SCORE_THREASHOLD', .7)
IMAGES_LOCATION = os.getenv('IMAGES_LOCATION')
AMOUNT_SIMILARITY_SEARCH_RESULTS = os.getenv('AMOUNT_SIMILARITY_SEARCH_RESULTS')

#overwrite FlashrankRerank compress_documents to keep metadata of original documents
def compress_documents(
    self,
    documents: Sequence[Document],
    query: str,
    callbacks: Optional[Callbacks] = None,
) -> Sequence[Document]:
    passages = [
        {"id": i, "text": doc.page_content} for i, doc in enumerate(documents)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    rerank_response = self.client.rerank(rerank_request)[: self.top_n]
    final_results = []
    for r in rerank_response:
        doc_index = r["id"]
        original_doc = documents[doc_index]
        
        metadata = {"id": r["id"], "relevance_score": r["score"]}
        metadata.update(original_doc.metadata)  # Merge metadata from original document
        
        doc = Document(
            page_content=r["text"],
            metadata=metadata,
        )
        final_results.append(doc)
    return final_results

FlashrankRerank.compress_documents=compress_documents


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


    def search_documents(self, query: str) -> List[Tuple[Document, float]]:
        """Searches the documents in the Qdrant DB with a specific query.

        Args:
            query (str): The question for which documents should be searched.

        Returns:
            List[Tuple[Document, float]]: A list of search results, where each result is a tuple
            containing a Document object and a float score.
        """
        amount = int(os.getenv("AMOUNT_SIMILARITY_SEARCH_RESULTS","10"))
        docs_with_score = self.vector_store.similarity_search_with_score(query, k=amount, score_threshold=SCORE_THREASHOLD)
        logger.debug(f"\nNumber of documents: {len(docs_with_score)}")

        if docs_with_score is not None and len(docs_with_score) > 0:
            logger.debug("SUCCESS: Documents found after similarity_search_with_score.")

            if ACTIVATE_RERANKER == "True":
                # Extract documents from the list of tuples i am not interessted in the scores anymore
                docs = [document for document, score in docs_with_score]

                logger.info(f"reranking ...")
                embedding = self.embedding_model
                
                retriever = self.vector_store.from_documents(docs, embedding, api_key=QDRANT_API_KEY, url=QDRANT_URL, collection_name="temp_ollama").as_retriever()
                ranker = Ranker(model_name="ms-marco-MultiBERT-L-12")
                rerank_compressor = FlashrankRerank(client= ranker, model="ms-marco-MultiBERT-L-12", top_n = 3)
                #rerank_compressor = CohereRerank(user_agent="my-app", model="rerank-multilingual-v2.0", top_n=3)
#                splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", "; ", "! ", "? ", "# "],chunk_size=120, chunk_overlap=20)
#                redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
                relevant_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=float(SCORE_THREASHOLD))
#                pipeline_compressor = DocumentCompressorPipeline(
#                    transformers=[splitter, redundant_filter, relevant_filter, rerank_compressor]
#                )

                pipeline_compressor = DocumentCompressorPipeline(
                    transformers=[relevant_filter, rerank_compressor]
                )
                compression_retriever1 = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
                reranked_docs = compression_retriever1.get_relevant_documents(query)

#                for document in docs:
#                    logger.info(f"Context after reranking: {replace_multiple_whitespaces(document.page_content)}")

                #Delete the temporary qdrant collection which is used for the base retriever
                url = f"{QDRANT_URL}:{QDRANT_PORT}/collections/temp_ollama"
                headers = {"Content-Type": "application/json", "api-key": QDRANT_API_KEY}
                requests.delete(url, headers=headers)

                logger.info(f"rerank document: {reranked_docs}")


                score_issueIds = get_issueid_dict(reranked_docs)

            else:
                # Extract issueIds with its highes score
                score_issueIds = get_issueid_score_dict(docs_with_score)


            #find documents by its issueId
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


            issue_docs = retriever.get_relevant_documents(query)
            logger.debug(f"\n {len(issue_docs)} filtered documents found")
            logger.info(f"Filtered docs {issue_docs} ")

            #Logic for none-reranking needs to be implemented here
            #filtered_docs = [t[0] for t in docs]
            return issue_docs
        
        
        return None

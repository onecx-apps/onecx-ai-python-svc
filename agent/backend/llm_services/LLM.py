from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
from langchain.docstore.document import Document as LangchainDocument
from omegaconf import DictConfig


class BaseLLM(ABC):
    @abstractmethod
    def summarize_text(self, text: str, cfg: DictConfig) -> str:
        """Summarizes the given text using the LLM API.

        Args:
            text (str): The text to be summarized.

        Returns:
            str: The summary of the text.
        """
        pass

    @abstractmethod
    def send_chat_completion(self, text: str, query: str, cfg: DictConfig, conversation_type: str, messages: any) -> str:
        """Sent completion request to LLM API.

        Args:
            text (str): The text on which the completion should be based.
            query (str): The query for the completion.
            cfg (DictConfig):

        Returns:
            str: Response from the LLM API.
        """
        pass

    @abstractmethod
    def chat(self, documents: list[tuple[LangchainDocument, float]], messages: any, query: str, conversation_type: str, summarization: bool = False) -> Tuple[str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]:
        """QA takes a list of documents and returns a list of answers.

        Args:
            documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
            query (str): The query to ask.
            summarization (bool, optional): Whether to use summarization. Defaults to False.

        Returns:
            Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
        """
        pass

    @abstractmethod
    def generate_request(self, url_ollama_generateEndpoint: str, model: str, full_prompt: str):
        pass


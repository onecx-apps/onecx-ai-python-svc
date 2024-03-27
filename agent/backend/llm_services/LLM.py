from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
from langchain.docstore.document import Document as LangchainDocument


class BaseLLM(ABC):

    @abstractmethod
    def chat(self, documents: list[tuple[LangchainDocument, float]], messages: any, query: str) -> Tuple[str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]:
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
    def generate(self, prompt: str) -> str:
        pass

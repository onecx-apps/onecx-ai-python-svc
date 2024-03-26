import os
import requests
from typing import Any, Dict, List, Tuple, Union
from dotenv import load_dotenv
from loguru import logger
from langchain.docstore.document import Document as LangchainDocument
from agent.utils.utility import replace_multiple_whitespaces, convert_message
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from agent.backend.llm_services.LLM import BaseLLM 
import ollama
from ollama import Client

client = Client(host='http://ollama.one-cx.org')


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mixtral")
OLLAMA_MODEL_VERSION = os.getenv('OLLAMA_MODEL_VERSION')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
Q_AND_A_SYS_MSG = os.getenv("Q_A_SYSTEM_MESSAGE",default="Du bist ein ehrlicher, respektvoller und ehrlicher Assistent. Zur Beantwortung der Frage nutzt du nur den Text, welcher zwischen <INPUT> und </INPUT> steht! Findest du keine Informationen im bereitgestellten Text, so antwortest du mit 'Ich habe dazu keine Informationen'")


class OllamaLLM(BaseLLM):
    def __init__(self):
        self.q_and_a_system_message = Q_AND_A_SYS_MSG
            

    def chat(self, documents: list[tuple[LangchainDocument, float]], messages: any, query: str) -> Tuple[str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]:
        """Takes a list of documents and returns a list of answers.

        Args:
            documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
            query (str): The query to ask.
            summarization (bool, optional): Whether to use summarization. Defaults to False.

        Returns:
            Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
        """
        text = ""
        # extract the texts and meta data from the documents
        texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
        text = " ".join(texts)
        meta_data = [doc.metadata for doc in documents]

        answer=""
        try:
            # fills the prompt, sends a request and returns the response
            #answer = self.send_chat_completion(text=text, query=query, conversation_type=conversation_type, messages=messages)

            ollama_model = f"{OLLAMA_MODEL}:{OLLAMA_MODEL_VERSION}" if OLLAMA_MODEL_VERSION else OLLAMA_MODEL


            # use the query and the found document texts and create a message with question and context
            message = convert_message(query, text)
            messages.append(message)

            llm_response = client.chat(model=ollama_model, 
                messages=messages,
                stream=False,
            )

            answer = llm_response['message']['content']


        except ValueError as e:
            logger.error("Error found:")
            logger.error(e)
            answer = "Error while processing the completion"
        logger.debug(f"LLM response: {answer}")
        
        return answer, meta_data






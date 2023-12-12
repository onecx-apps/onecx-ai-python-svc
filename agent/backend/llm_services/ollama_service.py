import os
from typing import Any, Dict, List, Tuple, Union
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from agent.utils.configuration import load_config
from agent.utils.utility import generate_prompt
from langchain.docstore.document import Document as LangchainDocument
import requests

from agent.utils.utility import replace_multiple_whitespaces
#from agent.backend.qdrant_service import get_qdrant_client

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import ChatOllama

from agent.backend.llm_services.LLM import BaseLLM  

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


class OllamaLLM(BaseLLM):
    def __init__(self):
        self.channeling_system_message = """Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung. Metadaten zu den einzelnen Quellen wie Autor, URL o.ä. sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der Quelle. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. Beantworte diese aus den Quellen. Sollten diese keine Antwort enthalten, antworte, dass auf Basis der gegebenen Informationen keine Antwort möglich ist! USER: BEGININPUT"""

        #self.q_and_a_system_message = """Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung. Metadaten zu den einzelnen Quellen wie Autor, URL o.ä. sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der Quelle. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. Beantworte diese aus den Quellen. Sollten diese keine Antwort enthalten, antworte, dass auf Basis der gegebenen Informationen keine Antwort möglich ist! USER: BEGININPUT"""

        #self.q_and_a_system_message = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Answer in the language you got asked."""

        self.q_and_a_system_message = os.getenv("Q_A_SYSTEM_MESSAGE",default="Du bist ein ehrlicher, respektvoller und ehrlicher Assistent. Zur Beantwortung der Frage nutzt du nur den Text, welcher zwischen <INPUT> und </INPUT> steht! Findest du keine Informationen im bereitgestellten Text, so antwortest du mit 'Ich habe dazu keine Informationen'")
        
    #just for tests or future summaries when the prompt gets too long
    @load_config(location="config/ai/llama2.yml")
    def summarize_text(self, text: str, cfg: DictConfig) -> str:
        """Summarizes the given text using the llama2 API.

        Args:
            text (str): The text to be summarized.

        Returns:
            str: The summary of the text.
        """
        prompt = generate_prompt(prompt_name=f"{OLLAMA_MODEL}-summarization.j2", text=text, language="de")

        ollama_model = f"{os.environ.get('OLLAMA_MODEL')}:{os.environ.get('OLLAMA_MODEL_VERSION')}" if os.environ.get('OLLAMA_MODEL_VERSION') else os.environ.get('OLLAMA_MODEL')

        llm = ChatOllama(
        base_url="http://" + OLLAMA_URL + ":" + OLLAMA_PORT,
        model=ollama_model,
        verbose=True
        )

        response = llm(prompt)

        return response

    @load_config(location="config/ai/llama2.yml")
    def send_chat_completion(self, text: str, query: str, cfg: DictConfig, conversation_type: str, messages: any) -> str:
        """Sent completion request to llama2 API.

        Args:
            text (str): The text on which the completion should be based.
            query (str): The query for the completion.
            cfg (DictConfig):

        Returns:
            str: Response from the llama2 API.
        """

        #fill the prompt if not q and a then it will use channeling with the documents
        if conversation_type == "CHANNELING":
            prompt = generate_prompt(prompt_name=f"{OLLAMA_MODEL}-channeling.j2", text=text, query=query, system=self.channeling_system_message, language="de")
        else:
            prompt = generate_prompt(prompt_name=f"{OLLAMA_MODEL}-qa.j2", text=text, query=query, system=self.q_and_a_system_message, language="de")
        messages.append({"role": "user", "content": prompt})
        logger.info(f"DEBUG: This is the filled prompt before request: {prompt}")

        
        ollama_model = f"{os.environ.get('OLLAMA_MODEL')}:{os.environ.get('OLLAMA_MODEL_VERSION')}" if os.environ.get('OLLAMA_MODEL_VERSION') else os.environ.get('OLLAMA_MODEL')

        # llm = ChatOllama(
        # base_url="http://" + OLLAMA_URL + ":" + OLLAMA_PORT,
        # model=ollama_model,
        # verbose=True
        # )

        messagesBaseFormat: List[BaseMessage] = [HumanMessage(content=m["content"], additional_kwargs={}) if m["role"] == "user"
                        else AIMessage(content=m["content"], additional_kwargs={}) if m["role"] == "assistant"
                        else SystemMessage(content=m["content"], additional_kwargs={})
                        for m in messages]

        raw_mode = os.environ.get('OLLAMA_RAW_MODE', default = "False").lower() in ['true']

        response = self.generate_request(url_ollama_generateEndpoint="http://ollama.one-cx.org:80/api/generate",
                                        model=os.environ.get('OLLAMA_MODEL'),
                                        full_prompt=prompt)

        # response = llm.generate(
        #     messages=[messagesBaseFormat],        
        # )
        
        logger.info(f"DEBUG: response: {response}")
        return response

    def chat(self, documents: list[tuple[LangchainDocument, float]], messages: any, query: str, conversation_type: str, summarization: bool = False) -> Tuple[str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]:
        """QA takes a list of documents and returns a list of answers.

        Args:
            documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
            query (str): The query to ask.
            summarization (bool, optional): Whether to use summarization. Defaults to False.

        Returns:
            Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
        """
        text = ""
        if conversation_type == "Q_AND_A":
            # if the list of documents contains only one document extract the text directly
            if len(documents) == 1:
                texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
                text = " ".join(texts)
                meta_data = [doc.metadata for doc in documents]

            else:
                # extract the text from the documents
                texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
                if summarization:
                    # call summarization
                    logger.info(f"woudl call a summary here")

                else:
                    # combine the texts to one text
                    text = " ".join(texts)
                meta_data = [doc.metadata for doc in documents]
        else:
            # if the list of documents contains only one document extract the text directly
            if len(documents) == 1:
                texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
                text = " ".join(texts)
                meta_data = [doc.metadata for doc in documents]

            else:
                # extract the text from the documents
                texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
                if summarization:
                    # call summarization
                    logger.info(f"woudl call a summary here")

                else:
                    # combine the texts to one text
                    text = " ".join(texts)
                meta_data = [doc.metadata for doc in documents]
        
        answer=""
        try:
            # call the gpt api

            answer = self.send_chat_completion(text=text, query=query, conversation_type=conversation_type, messages=messages)

        except ValueError as e:
            #when prompt is too large it can be implemented here
            logger.debug("DEBUG: Error found.")
            logger.error(e)
            answer = "Error"
        logger.debug(f"LLM response: {answer}")
        
        return answer, meta_data


    def generate_request(self, url_ollama_generateEndpoint: str, model: str, full_prompt: str):
        url = url_ollama_generateEndpoint
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "template": full_prompt,
            "stream": False,
            "options": {"stop": ["<|im_start|>", "<|im_end|>"]}
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            logger.debug("Request was successful!")
            logger.debug("Response:")
            logger.debug(response.json()["response"])
        else:
            logger.debug(f"Error {response.status_code}: {response.text}")

        return response.json()["response"]


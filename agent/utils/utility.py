"""This is the utility module."""
import os
import re
import uuid
from loguru import logger
from agent.backend.llm_services.LLM import BaseLLM
from agent.data_model.chatbot_model import Conversation, ChatMessage
from typing import List, Dict



def convert_messages(conversation: Conversation) -> List[Dict[str, str]]:
    messages = []
    for chat_message in conversation.history:
        role = chat_message.type.value
        content = chat_message.message
        messages.append({'role': role, 'content': content})

    return messages


def convert_message(query: str, context: str) -> Dict[str, str]:
    
    formatted_prompt = f"Question: {query}\n\nContext: {context}"
    role = "user"
    message = {'role': role, 'content': formatted_prompt}
    return message


def create_tmp_folder() -> str:
    """Creates a temporary folder for files to store.

    Returns:
        str: The directory name.
    """
    # Create a temporary folder to save the files
    tmp_dir = f"tmp_{str(uuid.uuid4())}"
    os.makedirs(tmp_dir)
    logger.info(f"Created new folder {tmp_dir}.")
    return tmp_dir



def replace_multiple_whitespaces(text):
    # Use regular expression to replace multiple whitespaces with a single whitespace
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text


if __name__ == "__main__":
    # test the function
    generate_prompt("qa.j2", "This is a test text.", "What is the meaning of life?")

def get_llm_service(name: str = "ollama") -> BaseLLM:
    if (name == "ollama"):
        from agent.backend.llm_services.ollama_service import OllamaLLM
        return OllamaLLM()
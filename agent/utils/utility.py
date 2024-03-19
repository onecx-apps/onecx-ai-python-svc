"""This is the utility module."""
import os
import re
import uuid
from jinja2 import Template
from loguru import logger
from agent.backend.llm_services.LLM import BaseLLM


def combine_text_from_list(input_list: list) -> str:
    """Combines all strings in a list to one string.

    Args:
        input_list (list): List of strings

    Raises:
        TypeError: Input list must contain only strings

    Returns:
        str: Combined string
    """
    # iterate through list and combine all strings to one
    combined_text = ""

    logger.info(f"List: {input_list}")

    for text in input_list:
        # verify that text is a string
        if isinstance(text, str):
            # combine the text in a new line
            combined_text += "\n".join(text)

        else:
            raise TypeError("Input list must contain only strings")

    return combined_text


def generate_prompt(prompt_name: str, text: str, query: str = "", system: str = "", language: str = "de") -> str:
    """Generates a prompt for the Luminous API using a Jinja template.

    Args:
        prompt_name (str): The name of the file containing the Jinja template.
        text (str): The text to be inserted into the template.
        query (str): The query to be inserted into the template.
        language (str): The language the query should output.

    Returns:
        str: The generated prompt.

    Raises:
        FileNotFoundError: If the specified prompt file cannot be found.
    """

    ### if False template of ollama is used, which is defined in Modelfile. otherwise it will use template .
    if os.environ.get('OLLAMA_RAW_MODE', default = "False") == "False":
        logger.info(f"DEBUG: using simple prompt text with context and query only ")
        prompt_text = " <INPUT> "+text+" </INPUT> "+query
    
    else:
        logger.info(f"DEBUG: using template for llm ")
        try:
            match language:
                case "en":
                    lang = "en"
                case "de":
                    lang = "de"
                case _:
                    raise ValueError("Language not supported.")
            with open(os.path.join("prompts", lang, prompt_name)) as f:
                prompt = Template(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file '{prompt_name}' not found.")

        # replace the value text with jinja
        # Render the template with your variable
        if query:
            prompt_text = prompt.render(text=text, query=query, system=system)
        else:
            prompt_text = prompt.render(text=text, system=system)

    #logger.info(f"DEBUG: This is the prompt after inserting: {prompt_text}")
    return prompt_text


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
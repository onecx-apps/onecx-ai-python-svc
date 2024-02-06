"""This is the utility module."""
import os
import re
import json
import uuid
from jinja2 import Template
from loguru import logger
from agent.backend.llm_services.LLM import BaseLLM
from langchain.docstore.document import Document

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


def get_token(token: str | None, llm_backend: str | None, aleph_alpha_key: str | None) -> str:
    """Get the token from the environment variables or the parameter.

    Args:
        token (str, optional): Token from the REST service.
        llm_backend (str): LLM provider.

    Returns:
        str: Token for the LLM Provider of choice.

    Raises:
        ValueError: If no token is provided.
    """
    env_token = aleph_alpha_key
    if not env_token and not token:
        raise ValueError("No token provided.")  #

    return token or env_token  # type: ignore


def validate_token(token: str | None, llm_backend: str, aleph_alpha_key: str | None) -> str:
    """Test if a token is available, and raise an error if it is missing when needed.

    Args:
        token (str): Token from the request
        llm_backend (str): Backend from the request
        aleph_alpha_key (str): Key from the .env file

    Returns:
        str: Token
    """

    token = get_token(token, llm_backend, aleph_alpha_key)

    return token


def extract_procedures_from_issue(documents, IMAGES_LOCATION):
    """
    Transform the given list of langchain_core Document objects into a new list of documents.

    Args:
        documents (list of Document): List of langchain_core Document objects.

    Returns:
        list: List of transformed langchain_core Document objects.
    """
    transformed_documents = []

    for doc in documents:

        pagecontent_json = json.loads(doc.page_content)

        # Create a new document for the main description
        main_doc = Document(
            metadata={
                "issueId": pagecontent_json["issueId"],
                "url": pagecontent_json["url"],
                "title": pagecontent_json["title"],
                "type": "issue",
                "source": doc.metadata.get('source', ''),
                "images":[]
            },
            page_content=pagecontent_json["title"] + "\n" + pagecontent_json["description"]
        )
        

        # Create documents for each procedure
        
        options_inc = 0
        image_urls = []
        for procedure in pagecontent_json["procedures"]:
            procedure_doc = Document(
                metadata={
                    "issueId": pagecontent_json["issueId"],
                    "url": procedure["url"],
                    "image_url": IMAGES_LOCATION + procedure["image_url"],
                    "type": "procedure"
                },
                page_content=procedure["name"] + "\n" + procedure["description"]
            )
            options_inc+=1
            main_doc.page_content = main_doc.page_content + "\nOption " + str(options_inc) + ":\n" + procedure["name"]
            image_urls.append(IMAGES_LOCATION + procedure["image_url"])
            transformed_documents.append(procedure_doc)


        main_doc.metadata["images"] = [{"image_url": url} for url in image_urls]
        transformed_documents.append(main_doc)

    return transformed_documents


def get_issueid_score_dict(documents):
    score_dict = {}

    # Loop through the list and update the dictionary with the highest score for each ID

    if documents is not None and len(documents) > 0:
        for element in documents:
            document, score = element
        
            current_id = document.metadata.get('issueId', '')
            current_score = score

            # Check if the ID is already in the dictionary
            if current_id in score_dict:
                # Update the score if the current score is higher
                if current_score > score_dict[current_id]:
                    score_dict[current_id] = current_score
            else:
                # Add the ID to the dictionary if not present
                score_dict[current_id] = current_score

    logger.info(f"SCORED_DICT {score_dict}.")


    # Sort the dictionary items based on scores
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    # Iterate through the sorted items to remove IDs with a too high difference
    filtered_dict = {}
    
    # Find the top score
    top_score = sorted_items[0][1]

    # Iterate through the sorted items to remove IDs with a too high difference compared to the top score
    filtered_dict = {}

    for current_id, current_score in sorted_items:
        if (top_score - current_score) <= 0.02:
            filtered_dict[current_id] = current_score
    
    logger.info(f"FILTERED_DICT {filtered_dict}.")
    return filtered_dict



def replace_multiple_whitespaces(text):
    # Use regular expression to replace multiple whitespaces with a single whitespace
    cleaned_text = re.sub(r'\s+', ' ', text)
    return '<CONTEXT>'+cleaned_text+'/<CONTEXT>\n'



# Function to add a solution to the structure
def add_solution(headline, image_urls, summary, url, json_data):
    solution = {
        "headline": headline,
        "images": image_urls,
        "summary": summary,
        "url": url,
    }
    json_data["solutions"].append(solution)


def get_llm_service(name: str = "ollama") -> BaseLLM:
    if (name == "ollama"):
        from agent.backend.llm_services.ollama_service import OllamaLLM
        return OllamaLLM()
    

def check_for_yes(input_string):
    # Define a regular expression pattern to match variations of "yes"
    yes_pattern = re.compile(r'\b(?:yes|ja)\b', re.IGNORECASE)

    # Use re.search to check if the pattern is present in the input string
    match = re.search(yes_pattern, input_string)

    # Return True if a match is found, otherwise return False
    return bool(match)


def detect_language(text):
    text = text.lower()

    if 'german' in text or 'deutsch' in text:
        return 'German'
    elif 'english' in text or 'englisch' in text:
        return 'English'
    elif 'french' in text or 'französisch' in text:
        return 'French'
    elif 'italian' in text or 'italienisch' in text:
        return 'Italian'
    elif 'spanish' in text or 'spanisch' in text:
        return 'Spanish'
    else:
        return 'English'



if __name__ == "__main__":
    # test the function
    generate_prompt("qa.j2", "This is a test text.", "What is the meaning of life?")

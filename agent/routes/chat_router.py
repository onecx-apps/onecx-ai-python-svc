from typing import Optional
import uuid
import time
import os
from fastapi import APIRouter, HTTPException, Body, Query
from ..data_model.chatbot_model import ChatMessage, Conversation, MessageType, ConversationType
import agent.data_model.response_model as Response
from loguru import logger
from agent.dependencies import document_service, llm
from agent.utils.utility import add_solution, check_for_yes, detect_language
import copy
import json

chat_router = APIRouter(tags=["chat"])
chatConversationMemory = []

def get_chat_by_conversation_id(conversationId):
    for conversation in chatConversationMemory:
        if conversation["conversationId"] == conversationId:
            return conversation
    return None

#function that returns a Conversation but without System Messages
def get_chat_by_conversation_id_filtered(conversationId):
    for conversation in chatConversationMemory:
        if conversation["conversationId"] == conversationId:
            conversationObj = Conversation(conversationId=conversation["conversationId"], history=conversation["history"], conversationType=ConversationType(conversation["conversationType"]))
            
            # Code to filter out the SYSTEM messages
            conversationObj.history = [msg for msg in conversationObj.history if msg.type != MessageType.SYSTEM]
            
            return conversationObj  # Return the filtered conversation object
    return None


@chat_router.get("/")
def read_root() -> str:
    """Returns the welcome message.

    Returns:
        str: The welcome message.
    """
    return "Welcome to the Chatbot-Backend!"


@chat_router.post("/chat")
async def chat_with_bot(chat_message: ChatMessage,
    use_llm: bool = Query(False, description="A flag indicating whether to generate responses using LLM")
) -> ChatMessage:
    # Check if conversation exists
    conversation = get_chat_by_conversation_id(chat_message.conversationId)

    # Convert ChatMessage to a dict format to append to history
    message_dict = chat_message.dict()
    message_dict["correlationId"] = str(uuid.uuid4())
    message_dictDTO = ChatMessage(conversationId=message_dict["conversationId"], correlationId=message_dict["correlationId"], message=message_dict["message"], type=message_dict["type"], creationDate=int(time.time()))

    if not conversation:
        # If conversation doesn't exist, raise an error
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # array for llama2 chat completion request - chat history with system message
    chatCompletionArr = []
    logger.debug(f"Here is conversation history:     {conversation['history']}")
    for msg in conversation["history"]:
        logger.debug(f"Here is msg:     {msg.type}")
        chatCompletionArr.append({"role": msg.type, "content": msg.message})

    # If conversation exists, append the message to its history
    conversation["history"].append(message_dictDTO)

    answer = ""
    language = ""

    analyize_question_prompt = f"""[INST] You are a helpful chat assistant. Analyze the provided TEXT and make a determination as to whether or not it is about a problem.
    TEXT: {message_dict["message"]}
    If it is a technical problem simply answer with precisely "YES" and the language of the TEXT. Do not add any additional explainations!
	A full sample response looks like:
	NO, French
    [/INST]
    """
    analyze_question_response = llm.generate(
        analyize_question_prompt) if use_llm else ""

    # Create an empty structure
    json_data = {
        "general_answer": "",
        "solutions": []
    }

    amount=int(os.getenv("AMOUNT_SIMILARITY_SEARCH_RESULTS","10"))
    documents = document_service.search_documents(query=message_dict["message"], amount=amount)

    if use_llm:
        language = detect_language(analyze_question_response)
        if not check_for_yes(analyze_question_response):
            answer_question_prompt = f"""[INST] You are a helpful chat assistant. Answer to the provided TEXT in {language} without any explainations!
            TEXT: {message_dict["message"]}
            [/INST]
            """
            answer_question_response = llm.generate(answer_question_prompt)
            json_data["general_answer"] = answer_question_response
        else:
            length_of_documents = len(documents) if documents is not None else 0
            general_response_prompt = f"""[INST] You are a helpful chat assistant. Tell in {language} that you found {length_of_documents} answers for the provided TEXT without any explainations and with one short sentence maximum!
            TEXT: {message_dict["message"]}
            [/INST]
            """
            general_answer = llm.generate(general_response_prompt)
            json_data["general_answer"] = general_answer
    else:
        json_data["general_answer"] = "Ich habe folgende mögliche Lösungen gefunden: \n\n"

    if documents:
        for document in documents:
            logger.info(f"document: {document}")

            if use_llm:
                #prompt = f"""[INST] I have the following TEXT *** {document.page_content} *** \n Please summarize this TEXT with maximum 3 sentences in the same language without any explainations. [/INST]"""
                prompt = f"""[INST] You are a helpful CONTENT summarization assistant. Your task is to generate a summarization text in {language} of the provided CONTENT :
                CONTENT: {document.page_content}
    
                Just generate the summarization without explanations.
                [/INST] """
                summary = llm.generate(prompt)
            else:
                summary = ""

            # Add the first solution
            add_solution(
                document.metadata.get('title', '---'),
                document.metadata.get('images', '[]'),
                summary,
                document.metadata.get('url', '---'),
                json_data
            )
    answer = json.dumps(json_data, ensure_ascii=False,  indent=4)
    logger.info(f"answers: {answer}")

    bot_response = ChatMessage(conversationId=chat_message.conversationId, correlationId=message_dict["correlationId"], message=answer, type=MessageType.ASSISTANT, creationDate=int(time.time()))
    conversation["history"].append(bot_response)

    return bot_response



@chat_router.get("/conversation/{conversationId}")
async def get_conversation(conversationId: str) -> Conversation:
    conversation = get_chat_by_conversation_id(conversationId)
    
    if  conversation == None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        return conversation

@chat_router.post("/startConversation")
async def start_conversation(conversation_type: str = Body(..., embed=True)) -> Conversation:
    conversation_id_uuid = str(uuid.uuid4())
    start_conversation = []
    if conversation_type == "Q_AND_A":
        iniialSystemMessage = ChatMessage(conversationId=conversation_id_uuid, correlationId="System Message", message=os.getenv("Q_A_SYSTEM_MESSAGE",default="Du bist ein ehrlicher, respektvoller und ehrlicher Assistent. Zur Beantwortung der Frage nutzt du nur den Text, welcher zwischen <INPUT> und </INPUT> steht! Findest du keine Informationen im bereitgestellten Text, so antwortest du mit 'Ich habe dazu keine Informationen'"), type=MessageType.SYSTEM, creationDate=int(time.time()))
        start_conversation.append(iniialSystemMessage)

        #If you want to add a welcome message into the message history enable the following:
        #welcomeMessage= ChatMessage(conversationId=conversation_id_uuid, correlationId="Welcome Message", message="Hallo ich bin dein Asisstent für heute! Was möchtest du wissen?(Q&A)", type=MessageType.ASSISTANT, creationDate=int(time.time()))
        #start_conversation.append(welcomeMessage)

    else:
        #Different system messages for each conversation type can be implemented here. E.g.: channeling.
        iniialSystemMessage = ChatMessage(conversationId=conversation_id_uuid, correlationId="System Message", message=os.getenv("Q_A_SYSTEM_MESSAGE",default="Du bist ein ehrlicher, respektvoller und ehrlicher Assistent. Zur Beantwortung der Frage nutzt du nur den Text, welcher zwischen <INPUT> und </INPUT> steht! Findest du keine Informationen im bereitgestellten Text, so antwortest du mit 'Ich habe dazu keine Informationen'"), type=MessageType.SYSTEM, creationDate=int(time.time()))
        start_conversation.append(iniialSystemMessage)

    chatConversationMemory.append({"conversationId": conversation_id_uuid, "history": start_conversation, "conversationType": conversation_type})

    conversation = get_chat_by_conversation_id(conversation_id_uuid)
    
    if  conversation == None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        return conversation
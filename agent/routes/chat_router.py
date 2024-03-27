from typing import Optional
import uuid
import time
import os
from fastapi import APIRouter, HTTPException
from ..data_model.chatbot_model import ChatMessage, Conversation, MessageType, ConversationType
from loguru import logger
from agent.dependencies import document_service, llm
from agent.utils.utility import check_for_yes, detect_language, create_message, convert_messages

chat_router = APIRouter(tags=["chat"])
chatConversationMemory = []


@chat_router.get("/")
def read_root() -> str:
    """Returns the welcome message.

    Returns:
        str: The welcome message.
    """
    return "Welcome to the Chatbot-Backend!"


@chat_router.post("/chat")
async def chat_with_bot(chat_message: ChatMessage, conversation: Conversation) -> ChatMessage:

    answer = ""
    language = ""

    analyize_question_prompt = f"""[INST] You are a helpful chat assistant. Analyze the provided TEXT and make a determination as to whether or not it is about a problem.
    TEXT: {chat_message.message}
    If it is a technical problem simply answer with precisely "YES" and the language of the TEXT. Do not add any additional explainations!
	A full sample response looks like:
	NO, French
    [/INST]
    """
    analyze_question_response = llm.generate(
        analyize_question_prompt)

    language = detect_language(analyze_question_response)
    if not check_for_yes(analyze_question_response):
        answer_question_prompt = f"""[INST] You are a helpful chat assistant. Answer to the provided TEXT in {language} without any explainations!
        TEXT: {chat_message.message}
        [/INST]
        """
        answer_question_response = llm.generate(answer_question_prompt)
        botResponse = ChatMessage(conversationId=conversation.conversationId, message=answer_question_response, type=MessageType.ASSISTANT, creationDate=int(time.time()))
        return botResponse
    else:
        ollama_messages = []

        # Convert Conversation Messages to a dict format to append to history
        messages = convert_messages(conversation)
        ollama_messages.append(messages)
        
        language_system_message = create_message("system", f"Answer in {language}")
        ollama_messages[0].append(language_system_message)

        print(ollama_messages)

        #response bot
        documents = document_service.search_documents(query=chat_message.message)
        answer, meta_data = llm.chat(query=chat_message.message, documents=documents, messages=ollama_messages)

        
        botResponse = ChatMessage(conversationId=conversation.conversationId, message=answer, type=MessageType.ASSISTANT, creationDate=int(time.time()))
        return botResponse




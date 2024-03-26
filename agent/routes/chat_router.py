from typing import Optional
import uuid
import time
import os
from fastapi import APIRouter, HTTPException
from ..data_model.chatbot_model import ChatMessage, Conversation, MessageType, ConversationType
from loguru import logger
from agent.dependencies import document_service, llm
from agent.utils.utility import convert_messages, convert_message

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

    ollama_messages = []

    # Convert Conversation Messages to a dict format to append to history
    messages = convert_messages(conversation)
 
    ollama_messages.append(messages)
    

    #response bot
    documents = document_service.search_documents(query=chat_message.message)
    answer, meta_data = llm.chat(query=chat_message.message, documents=documents, messages=ollama_messages)

    
    botResponse = ChatMessage(conversationId=conversation.conversationId, message=answer, type=MessageType.ASSISTANT, creationDate=int(time.time()))
    return botResponse




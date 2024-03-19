from typing import Optional
import uuid
import time
import os
from fastapi import APIRouter, HTTPException, Body
from ..data_model.chatbot_model import ChatMessage, Conversation, MessageType, ConversationType
import agent.data_model.response_model as Response
from loguru import logger
from agent.dependencies import document_service, llm


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
async def chat_with_bot(chat_message: ChatMessage, conversation: Optional[Conversation] = None) -> ChatMessage:

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

    #response bot
    documents = document_service.search_documents(query=message_dict["message"], amount=os.getenv("AMOUNT_SIMILARITY_SEARCH_RESULTS",10))
    answer, meta_data = llm.chat(query=message_dict["message"], documents=documents, conversation_type=conversation["conversationType"], messages=chatCompletionArr)
    botResponse = ChatMessage(conversationId=chat_message.conversationId, correlationId=message_dict["correlationId"], message=answer, type=MessageType.ASSISTANT, creationDate=int(time.time()))
    conversation["history"].append(botResponse)
    return botResponse

@chat_router.get("/conversation/{conversationId}")
async def get_conversation(conversationId: str) -> Conversation:
    conversation = get_chat_by_conversation_id(conversationId)
    
    if  conversation == None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        return conversation


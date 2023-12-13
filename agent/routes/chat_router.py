import uuid
import time
import os
from fastapi import APIRouter, HTTPException, Body
from ..data_model.chatbot_model import ChatMessageDTO, ConversationDTO, MessageType, ConversationType
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

#function that returns a ConversationDTO but without System Messages
def get_chat_by_conversation_id_filtered(conversationId):
    for conversation in chatConversationMemory:
        if conversation["conversationId"] == conversationId:
            conversationObj = ConversationDTO(conversationId=conversation["conversationId"], history=conversation["history"], conversationType=ConversationType(conversation["conversationType"]))
            
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
    return "Welcome to the BMI Chatbot Backend!"


@chat_router.post("/chat")
async def chat_with_bot(chat_message: ChatMessageDTO) -> ChatMessageDTO:
    # Check if conversation exists
    conversation = get_chat_by_conversation_id(chat_message.conversationId)
    
    # Convert ChatMessageDTO to a dict format to append to history
    message_dict = chat_message.dict()
    message_dict["correlationId"] = str(uuid.uuid4())
    message_dictDTO = ChatMessageDTO(conversationId=message_dict["conversationId"], correlationId=message_dict["correlationId"], message=message_dict["message"], type=message_dict["type"], creationDate=int(time.time()))

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
    botResponse = ChatMessageDTO(conversationId=chat_message.conversationId, correlationId=message_dict["correlationId"], message=answer, type=MessageType.ASSISTANT, creationDate=int(time.time()))
    conversation["history"].append(botResponse)
    return botResponse

@chat_router.get("/conversation/{conversationId}")
async def get_conversation(conversationId: str) -> ConversationDTO:
    # Implement your logic here

    conversation = get_chat_by_conversation_id_filtered(conversationId)
    
    if  conversation == None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        return conversation

@chat_router.post("/startConversation")
async def start_conversation(conversation_type: str = Body(..., embed=True)) -> ConversationDTO:
    conversation_id_uuid = str(uuid.uuid4())
    start_conversation = []
    if conversation_type == "CHANNELING":
        # iniialSystemMessage = ChatMessageDTO(conversationId=conversation_id_uuid, correlationId="System Message", message=channeling_system_message, type=MessageType.SYSTEM, creationDate=int(time.time()))
        # start_conversation.append(iniialSystemMessage)
        welcomeMessage= ChatMessageDTO(conversationId=conversation_id_uuid, correlationId="System Message", message="Hallo ich bin dein Asisstent und führe dich durch den Anmeldeprozess. Wie ist dein Name?", type=MessageType.ASSISTANT, creationDate=int(time.time()))
        start_conversation.append(welcomeMessage)
    if conversation_type == "Q_AND_A":
        # iniialSystemMessage = ChatMessageDTO(conversationId=conversation_id_uuid, correlationId="System Message", message=q_and_a_system_message, type=MessageType.SYSTEM, creationDate=int(time.time()))
        # start_conversation.append(iniialSystemMessage)
        welcomeMessage= ChatMessageDTO(conversationId=conversation_id_uuid, correlationId="System Message", message="Hallo ich bin dein Asisstent für heute! Was möchtest du wissen?(Q&A)", type=MessageType.ASSISTANT, creationDate=int(time.time()))
        start_conversation.append(welcomeMessage)

    chatConversationMemory.append({"conversationId": conversation_id_uuid, "history": start_conversation, "conversationType": conversation_type})

    conversation = get_chat_by_conversation_id_filtered(conversation_id_uuid)
    
    if  conversation == None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        return conversation
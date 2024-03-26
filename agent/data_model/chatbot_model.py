"""Script that contains the Pydantic Models for the Rest Request."""

from typing import List, Optional
from fastapi import UploadFile
from pydantic import BaseModel, Field
from enum import Enum

class MessageType(str, Enum):
    ASSISTANT = "assistant"
    USER = "user"
    ACTION = "action"
    SYSTEM = "system"

class ConversationType(str, Enum):
    Q_AND_A = "Q_AND_A"

class ChatMessage(BaseModel):
    conversationId: str = Field(None, title="conversationId", description="The unique id for the whole user converstation.")
    message: str = Field(None, description="the message", max_length=10000)
    type: MessageType = Field(None, description="The enum message type")
    creationDate: int = Field(None, description="The timestamp in milliseconds")

class Conversation(BaseModel):
    conversationId: str = Field(None, title="conversationId", description="The unique id for the whole user converstation.")
    history: List[ChatMessage] = Field(None, description="A list of chat messages")
    conversationType: ConversationType = Field(None, description="The enum message type")

class Document(BaseModel):
    content: str = Field(None, description="Base64 encoded document content")

class UploadFile(BaseModel):
    filename: str = Field(None, description="Filename of the received file")
    size: int = Field(None, description="Filesize in Bytes")
    content_type: str = Field(None, description="MIME Type / media type")
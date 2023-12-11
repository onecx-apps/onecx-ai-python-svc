from pydantic import BaseModel, Field
import enum

class SimpleDetailMessage(BaseModel):
    detail: str = Field(None, description="response message", max_length=10000)
    
# Additional Responses
NOT_FOUND = {"model": SimpleDetailMessage, "description": "Requested or used item not found."}
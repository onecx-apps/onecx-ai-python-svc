from agent.backend.document_service import DocumentService
from agent.utils.utility import get_llm_service

document_service = DocumentService()
llm = get_llm_service()
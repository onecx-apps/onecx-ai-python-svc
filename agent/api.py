"""FastAPI Backend for the Knowledge Agent."""
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from loguru import logger
from agent.backend.document_service import DocumentService
from .routes import chat_router

# add file logger for loguru
logger.add("logs/file_{time}.log", backtrace=False, diagnose=False)
logger.info("Startup.")

# initialize the Fast API Application.
app = FastAPI(debug=True)
load_dotenv()

logger.info("Loading REST API Finished.")

# Include Routers
app.include_router(chat_router.chat_router)

@app.get("/")
def read_root() -> str:
    """Returns the welcome message.

    Returns:
        str: The welcome message.
    """
    return "Welcome to the BMI Chatbot Backend!"

# Add OpenAPI Generation Defines
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Chatbot Rest Api",
        version="1.0.1",
        description="This is the scheme for the **chatbot rest api**",
        routes=app.routes,
        tags=[{
                "name": "chat",
                "description": "All chat operations can be found organized under tag **chat**"
               },
              {
                "name": "document",
                "description": "All document operations can be found organized under tag **document**"
               }]
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9d/Capgemini_201x_logo.svg"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

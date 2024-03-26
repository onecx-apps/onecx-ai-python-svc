<br />
<div align="center">
  <a href="https://github.com/onecx-apps/onecx-ai-svc">
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/9d/Capgemini_201x_logo.svg" alt="Logo" width="200px">
  </a>

  <h3 align="center">REST Backend for a Conversational Chat Agent</h3>

  <p align="center">
    An advanced Retrieval Augmented Generation PoC solution for a GenAi Chatbot!
    <br />
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Project-Description">Description</a></li>
    <li><a href="#RAG-Architecture">Architecture</a></li>
    <li>
      <a href="#quickstart">Setup</a>
    </li>
    <li><a href="#API-Documentation">API Docs</a></li>
    <li><a href="#Example-Workflow">Example</a></li>
    <li><a href="#Roadmap-for-future-improvements">Roadmap</a></li>
  </ol>
</details>


## Project Description
This project is a versatile chatbot system designed to integrate seamlessly with various Large Language Models (LLM). Leveraging an advanced Retrieval-Augmented Generation (RAG) architecture, it empowers developers to build intelligent and context-aware chatbots.

Features:
- Q and A over documents
- Provides a REST API built with FastAPI for easy integration with other applications.
- Vector Storage with Qdrant.
- Advanced RAG methods like reranking and small to big chunking.

Technology Keywords:
- Python
- Ollama
- LangChain
- HuggingFace
- Embedding Model
- Reranking Model
- FastAPI
- uvicorn
- Qdrant
- Jinja Templating.
- Docker


## RAG Architecture
<div align="center">
  <img src="resources/RAG_diagramm.svg" alt="architecture">
</div>



## Quickstart

First of all you have to clone the repo, rename the env-template file into ".env" and fill up the missing values.<br />
Feel free to ask for support regarding the .env: @davidatcap & @michaelgloeckner<br />
<br />
From now on we will assume that this repo will be cloned onto a linux system.<br />
We will also assume that python 3.10 is already installed.<br />
Its up to you to use a virtual enviroment like conda for this project.<br />
<br />
You can also use the provided docker setup instead of a directly local setup.<br />
<br />

For the local setup lets start by installing the python packages on your system or enviroment:
```bash
pip install -r requirements.txt
```

Now get the local vector storage running (as a container preferred):
```bash
docker compose  -f "docker-compose.yml" up -d --build qdrant 
```

If you want to use a LLM locally you will have to install Ollama and a model.<br />
Ollama provides an interface which allows you to pull and run several LLMs locally with an REST API.<br />
Please visit https://ollama.ai/download and follow their instructions.<br />
This is how you can run a model with Ollama e.g.:
```bash
ollama run llama2
```

Now should be the vector storage and the LLMs running.<br />
Once these 2 services are running we can go on and start the backend service.
```bash
uvicorn agent.api:app --host 0.0.0.0 --port 8001

```
## API Documentation
Then you can go to http://127.0.0.1:8001/docs or http://127.0.0.1:8001/redoc to see the API documentation.<br />
The vector storage can be checked here: http://localhost:6333/dashboard

## Example Workflow
Start a conversation by requesting the /startConversation endpoint with the conversation type "Q_AND_A" in the body.
Then save the conversation-ID and paste it into the next request to the endpoint /chat.

<!-- ROADMAP -->
## Roadmap for future improvements:

- [ ] Combining BM25 Keyword Search and the semantic search results before reranking for diverse context results
- [ ] Summarizing retrieved context with a LLM call to get the conext more precise
- [ ] Summarizing message history with a LLM call when it gets too long
- [ ] Each conversation needs to be able to search over their individual documents
- [ ] LLM Result Evaluation
- [ ] Finding a solid open source reranker alternative to cohere
- [ ] Implement other document importers to receive data from e.g. confluence or scraping data from websites




{
  "chat_message": {
    "conversationId": "string",
    "correlationId": "string",
    "message": "Tell me a joke",
    "type": "user",
    "creationDate": 0
  },
  "conversation": {
    "conversationId": "string",
    "history": [
      {
        "conversationId": "string",
        "correlationId": "string",
        "message": "Talk like luigi",
        "type": "system",
        "creationDate": 0
      }
    ],
    "conversationType": "Q_AND_A"
  }
}

# Conversational Agent
This is a Rest-Backend for a Conversational Agent, that allows to embedd Documentes, search for them using Semantic Search, to QA based on Documents and do document processing with Large Language Models.


- [Conversational Agent](#conversational-agent)
  - [Recent Updates](#recent-updates)
  - [Quickstart](#quickstart)
  - [Redoc API Documentation](#redoc-api-documentation)
  - [Project Description](#project-description)
  - [Semantic Search](#semantic-search)
  - [Architecture](#architecture)
  - [Components](#components)
  - [Available LLM Backends](#available-llm-backends)
  - [Deployment](#deployment)
  - [Secret Management](#secret-management)
  - [Installation \& Development Backend](#installation--development-backend)
  - [Development Frontend](#development-frontend)
  - [Vector Database](#vector-database)
  - [Qdrant API Key](#qdrant-api-key)


## Recent Updates
- Added Qdrant as a vector database.


## Quickstart



To run the complete system with docker use this command:

```bash
git clone https://github.com/onecx-apps/onecx-ai-svc.git
cd onecx-ai-svc
```
Create a .env file from the .env-template and set the qdrant api key. For tests just set it to test.
QDRANT_API_KEY="test"

Then start the system with
```bash
  docker compose up -d
```

Then go to http://127.0.0.1:8001/docs or http://127.0.0.1:8001/redoc to see the API documentation.


## Docker build

docker build . -t onecx-ai-svc


## Redoc API Documentation
<!-- # TODO: -->



## Project Description
This project is a conversational agent that uses Aleph Alpha Large Language Model to generate responses to user queries. The agent also includes a vector database and a REST API built with FastAPI.

Features
- Uses Aleph Alpha Large Language Model to generate responses to user queries.
- Includes a vector database to store and retrieve information.
- Provides a REST API built with FastAPI for easy integration with other applications.
- Has a basic gui.

## Semantic Search
![Semantic Search Architecture](resources/search_flow.png)

Semantic search is an advanced search technique that aims to understand the meaning and context of a user's query, rather than matching keywords. It involves natural language processing (NLP) and machine learning algorithms to analyze and interpret user intent, synonyms, relationships between words, and the structure of content. By considering these factors, semantic search improves the accuracy and relevance of search results, providing a more intuitive and personalized user experience.

## Architecture
![Semantic Search Architecture](resources/Architecture.png)

## Components

Langchain is a library for natural language processing and machine learning. FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. A Vectordatabase is a database that stores vectors, which can be used for similarity searches and other machine learning tasks.

## Available LLM Backends

- [Aleph Alpha Luminous](https://aleph-alpha.com/)

## Deployment

If you want to use a default token for the LLM Provider you need to create a .env file. Do this by copying the .env-template file and add the necessary api keys.

If you want to build the image on your local machine you can use this command:

```bash
docker compose up
```

## Secret Management

Two ways to manage your api keys are available, the easiest approach is to sent the api token in the request as the token.
Another possiblity is to create a .env file and add the api token there.


## Development Frontend

To run the Frontend use this command in the root directory of the agent container:

```bash
poetry run streamlit run gui.py
```

## Vector Database

Qdrant Dashboard is available at http://QDRANT_URL:QDRANT_PORT/dashboard. There you need to enter the api key.


## Qdrant API Key
To use the Qdrant API you need to set the correct parameters in the .env file.
QDRANT_API_KEY is the API key for the Qdrant API in case you use external qdrant
QDRANT_URL the url of the qdrant
QDRANT_PORT the http port of qdrant
And you need to change it in the qdrant.yaml file in the config folder.

## Import documents into qdrant from bucket
GOOGLE_APPLICATION_CREDENTIALS_BASE64=
DOCUMENTS_BUCKET=bmi-immigration
call https://localhost:8001/embeddings/importDocuments and import documents which are in the DOCUMENTS_BUCKET bucket.

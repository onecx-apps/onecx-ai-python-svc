import os
from langchain.vectorstores import Qdrant
from langchain.schema.embeddings import Embeddings
from qdrant_client import QdrantClient, models
from agent.utils.configuration import load_config
from loguru import logger
from omegaconf import DictConfig

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

@load_config(location="config/db.yml")
def get_db_connection(cfg: DictConfig, embedding_model: Embeddings) -> Qdrant:
    """get_db_connection initializes the connection to the Qdrant db.

    :param cfg: OmegaConf configuration
    :type cfg: DictConfig
    :return: Qdrant DB connection
    :rtype: Qdrant
    """
    embedding = embedding_model
    qdrant_client = QdrantClient(os.getenv("QDRANT_URL",cfg.qdrant.url), port=os.getenv("QDRANT_PORT",cfg.qdrant.port), api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=cfg.qdrant.prefer_grpc)
    try: 
        qdrant_client.get_collection(collection_name=cfg.qdrant.collection_name)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=cfg.qdrant.collection_name,
            vectors_config=models.VectorParams(size=len(embedding.embed_query("Test text")), distance=models.Distance.COSINE),
        )
        logger.info(f"SUCCESS: Collection {cfg.qdrant.collection_name} created.")
    vector_db = Qdrant(client=qdrant_client, collection_name=cfg.qdrant.collection_name, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB Connection.")
    return vector_db
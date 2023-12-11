from loguru import logger
from typing import Callable
from .aws_cloud_service import AWSCloudService
from .google_cloud_service import GoogleCloudService

class CloudServiceFactory():
    _registry = {}
    _instance = {}
    
    def __new__(cls, name: str):
        if not "aws" in cls._registry:
            cls._registry["aws"] = AWSCloudService
        if not "gcs" in cls._registry:
            cls._registry["gcs"] = GoogleCloudService
        if not name.lower() in cls._registry:
            logger.error("The specified cloud provider is not available. Please set env variable CLOUD_PROVIDER correctly.")
        if not name.lower() in cls._instance:
            cls._instance[name.lower()] = cls._registry[name.lower()]()
        
        return cls._instance[name.lower()]
    
    @classmethod
    def _register_subclass(cls, name: str) -> Callable:
        def inner(wrapped_class) -> Callable:
            if name.lower() in cls._registry:
                logger.warning(f"CloudService {name.lower()} already exists and will be replaced.")
            cls._registry[name.lower()] = wrapped_class
            return wrapped_class
        return inner
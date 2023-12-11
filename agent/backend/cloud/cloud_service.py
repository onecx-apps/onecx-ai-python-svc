from abc import ABC, abstractmethod

class CloudService(ABC):
    @abstractmethod
    def download_files_from_bucket(self) -> None:
        pass
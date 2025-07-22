from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    mongo_db_name: str
    mongo_collection_name: str
    file_name: str


@dataclass(frozen=True)
class DataProcessingConfig:
    root_dir: Path
    preprocessor_object_file_name: str

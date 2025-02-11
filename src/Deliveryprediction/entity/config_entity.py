from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataCleaningConfig:
  root_dir: Path
  data_input_dir: Path
  preprocess_data_dir: Path

@dataclass(frozen=True)
class DataPreparationConfig:
    root_dir: Path
    data_input_dir: Path
    train_dir: Path
    test_dir: Path
    params : dict
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


@dataclass(frozen=True)
class DataTransformerConfig:
    root_dir: Path
    data_input_dir: Path
    data_tran_dir: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_input_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    min_child_weight: int
    min_split_gain: float
    reg_lambda: float
    n_jobs: int
    n_estimators_rf: int
    max_depth_rf: int
    criterion_rf: str
    max_features_rf: int
    min_samples_split_rf: int
    min_samples_leaf_rf: int
    max_samples_rf: float
    verbose_rf: int
    n_jobs_rf: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_input_dir:Path
    model_path: Path
    metric_file: Path
    all_params: dict
    prepro_dir: Path
    #metric_file_name: Path
    #target_column: str

@dataclass(frozen=True)
class ModelRegisterConfig:
  dagshu_tracking_uri : Path
  run_json_info: Path
  repo_owner: str
  repo_name: str    
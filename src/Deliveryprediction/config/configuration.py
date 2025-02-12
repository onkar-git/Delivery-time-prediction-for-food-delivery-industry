from Deliveryprediction.constants import *
from Deliveryprediction.utils.common import read_yaml, create_directories
from Deliveryprediction.entity.config_entity import (DataIngestionConfig,
                                                     DataValidationConfig,
                                                     DataCleaningConfig,
                                                     DataPreparationConfig,
                                                     DataTransformerConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    
    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config.data_cleaning

        create_directories([config.root_dir])

        data_cleaning_config = DataCleaningConfig(
            root_dir = config.root_dir,
            data_input_dir = config.data_input_dir,
            preprocess_data_dir = config.preprocess_data_dir
           )

        return data_cleaning_config


    def get_data_preparation_config(self) -> DataPreparationConfig:
            config = self.config.data_preparation
            params = self.params.PARAMS

            create_directories([config.root_dir])

            data_Preparation_config = DataPreparationConfig(
                root_dir = config.root_dir,
                data_input_dir = config.data_input_dir,
                train_dir = config.train_dir,
                test_dir = config.test_dir,
                params = params    
            )

            return data_Preparation_config
    

    def get_data_trans_config(self) -> DataTransformerConfig:
        config = self.config.data_transformation
        

        create_directories([config.root_dir])

        data_Transformation_config = DataTransformerConfig(
            root_dir = config.root_dir,
            data_input_dir = config.data_input_dir,
            data_tran_dir = config.data_tran_dir
        )

        return data_Transformation_config
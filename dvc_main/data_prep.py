import numpy as np
import pandas as pd
from pathlib import Path
from Deliveryprediction import logger
from Deliveryprediction.entity.config_entity import DataCleaningConfig,DataPreparationConfig,DataTransformerConfig,ModelTrainerConfig
from Deliveryprediction.config.configuration import ConfigurationManager
from Deliveryprediction.components._3_data_cleaning import DataCleaning
from Deliveryprediction.components.step_4_data_preparation import DataPreparation
from Deliveryprediction.components.step_5_data_transformation import DataTransformation
from Deliveryprediction.components.step_6_model_training import ModelTrainer 

try:
    logger.info("==Data Preparation Stage tracjking with DVC====")
    with open(Path("artifacts/data_validation/status.txt"), "r") as f:
        status = f.read().split(" ")[-1]

    if status == "True":
        config = ConfigurationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        data=data_preparation.load_data()
        data_preparation.split_data(data)
        logger.info("== train-test splitting Data Preparation Stage completed and tracking with DVC====")
    else:
        raise Exception("You data schema is not valid")

except Exception as e:
    print(e)

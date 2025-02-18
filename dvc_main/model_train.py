
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
    with open(Path("artifacts/data_validation/status.txt"), "r") as f:
        status = f.read().split(" ")[-1]

    if status == "True":
        config = ConfigurationManager()
        model_trainer = config.get_model_gbm_config()
        model_final = ModelTrainer(model_trainer)
        model_final.run()
    else:
        raise Exception("You data schema is not valid")

except Exception as e:
    print(e)
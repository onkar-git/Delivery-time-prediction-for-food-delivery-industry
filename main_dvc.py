import numpy as np
import pandas as pd
from pathlib import Path
from Deliveryprediction import logger
from Deliveryprediction.entity.config_entity import DataCleaningConfig,DataPreparationConfig,DataTransformerConfig
from Deliveryprediction.config.configuration import ConfigurationManager
from Deliveryprediction.components._3_data_cleaning import DataCleaning
from Deliveryprediction.components.step_4_data_preparation import DataPreparation
from Deliveryprediction.components.step_5_data_transformation import DataTransformation

try:
    logger.info("Data cleaning tracking with DVC manager")
    with open(Path("E://projects//Delivery-time-prediction-for-food-devlivery-industry//artifacts//data_validation//status.txt"), "r") as f:
        status = f.read().split(" ")[-1]

    if status == "True":
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config()
        cleaning_data = DataCleaning(config=data_cleaning_config)
        data=cleaning_data.load_data()
        cleaning_data.perform_data_cleaning(data)
        
    else:
        raise Exception("You data schema is not valid check artifact of data valiadtion")

except Exception as e:
    print(e)


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

try:
    logger.info("==Data Preparation Stage tracjking with DVC====")
    with open(Path("artifacts/data_validation/status.txt"), "r") as f:
        status = f.read().split(" ")[-1]

    if status == "True":
        config = ConfigurationManager()
        data_Transformation_config = config.get_data_trans_config()
        data_Transformation = DataTransformation(config=data_Transformation_config)
        data_Transformation.run_transformation_pipeline()
        logger.info("== train-test splitting Data Preparation Stage completed and tracking with DVC====")
    else:
        raise Exception("You data schema is not valid")

except Exception as e:
        print(e)
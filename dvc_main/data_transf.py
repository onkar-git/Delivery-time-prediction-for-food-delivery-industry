import numpy as np
import pandas as pd
from pathlib import Path
from Deliveryprediction import logger
from Deliveryprediction.entity.config_entity import DataCleaningConfig,DataPreparationConfig,DataTransformerConfig,ModelTrainerConfig
from Deliveryprediction.config.configuration import ConfigurationManager
from Deliveryprediction.components.step_5_data_transformation import DataTransformation


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
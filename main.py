
from Deliveryprediction import logger
from Deliveryprediction.pipeline.stage_01_data_ingetion import DataIngestionTrainingPipeline
from Deliveryprediction.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from Deliveryprediction.pipeline.stage_03_data_cleaning import DataCleaningPipeline
from Deliveryprediction.pipeline.stage_04_data_preparation import DataPreparationTrainingPipeline
from Deliveryprediction.pipeline.stage_05_data_transformation import DataTransformerTrainingPipeline
from Deliveryprediction.pipeline.stage_06_model_traning import ModelTrainerTrainingPipeline
from pathlib import Path

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>> Stage {STAGE_NAME} Started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<\n\nx===============")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation stage"

try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Cleaning Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    with open(Path("E://projects//Delivery-time-prediction-for-food-devlivery-industry//artifacts//data_validation//status.txt"), "r") as f:
        status = f.read().split(" ")[-1]

    if status == "True":
       clean_obj= DataCleaningPipeline()
       clean_obj.main()   
    else:
        raise Exception("You data schema is not valid check artifact of data valiadtion")
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    print(e)


STAGE_NAME = "Data Praparation Stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_preparation = DataPreparationTrainingPipeline()
   data_preparation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation Stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_Transformation = DataTransformerTrainingPipeline()
   data_Transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Training Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_train = ModelTrainerTrainingPipeline()
   model_train.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
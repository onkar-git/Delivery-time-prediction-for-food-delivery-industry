from Deliveryprediction.config.configuration import ConfigurationManager
from Deliveryprediction.components.step_6_model_training import ModelTrainer
from Deliveryprediction import logger
from pathlib import Path

STAGE_NAME = "Data Transformtion stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
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
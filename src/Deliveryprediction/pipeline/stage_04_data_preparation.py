from Deliveryprediction.config.configuration import ConfigurationManager
from Deliveryprediction.components.step_4_data_preparation import DataPreparation
from Deliveryprediction import logger
from pathlib import Path

STAGE_NAME = "Data Preparation stage"

class DataPreparationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_preparation_config = config.get_data_preparation_config()
                data_preparation = DataPreparation(config=data_preparation_config)
                data=data_preparation.load_data()
                data_preparation.split_data(data)
            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)
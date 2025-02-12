from Deliveryprediction.config.configuration import ConfigurationManager
from Deliveryprediction.components.step_5_data_transformation import DataTransformation
from Deliveryprediction import logger
from pathlib import Path

STAGE_NAME = "Data Transformtion stage"

class DataTransformerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_Transformation_config = config.get_data_trans_config()
                data_Transformation = DataTransformation(config=data_Transformation_config)
                data_Transformation.run_transformation_pipeline()
            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)
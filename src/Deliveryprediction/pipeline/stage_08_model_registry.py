from Deliveryprediction.config.configuration import ConfigurationManager
from Deliveryprediction.components.step_8_model_registry import MLFlowModelRegistry
from Deliveryprediction import logger
from pathlib import Path

STAGE_NAME = "Model registration stage"

class ModelRegistryTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                model_registry=config.get_model_registry_config()
                model_register =MLFlowModelRegistry(logger=logger,config=model_registry)
                model_register.register_and_transition_model()
            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e) 
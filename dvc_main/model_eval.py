from Deliveryprediction.config.configuration import ConfigurationManager
from Deliveryprediction.components.step_7_model_evaluation import ModelEvaluation
from Deliveryprediction import logger
from pathlib import Path

STAGE_NAME = "Data Transformtion stage"


try:
    with open(Path("artifacts/data_validation/status.txt"), "r") as f:
        status = f.read().split(" ")[-1]

    if status == "True":
        config = ConfigurationManager()
        model_config = config.get_model_evaluation_config()
        model_final = ModelEvaluation(logger=logger,experiment_name= 'delivery_prediction_experiemnt_1', repo_owner='onkar-git', repo_name='Delivery-time-prediction-for-food-delivery-industry', config=model_config)
        model_final.run()
    else:
        raise Exception("You data schema is not valid")

except Exception as e:
    print(e)
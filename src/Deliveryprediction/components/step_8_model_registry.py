import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
from Deliveryprediction import logger
from Deliveryprediction.entity.config_entity import ModelRegisterConfig

class MLFlowModelRegistry:
    def __init__(self,logger,config=ModelRegisterConfig):
        self.config = config
        self.repo_owner = config.repo_owner
        self.repo_name = config.repo_name
        self.logger = logger
        self.tracking_uri = config.dagshu_tracking_uri
        self.run_info_path = Path(config.run_json_info)
        self._initialize_dagshub()
        self._set_tracking_uri()
        self.client = MlflowClient()
        

    def _initialize_dagshub(self):
        dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)

    def _set_tracking_uri(self):
        mlflow.set_tracking_uri(self.tracking_uri)

    def load_model_information(self):
        with open(self.run_info_path) as f:
            return json.load(f)

    def register_and_transition_model(self):
        run_info = self.load_model_information()
        run_id = run_info["run_id"]
        model_name = run_info["model_name"]
        
        model_registry_path = f"runs:/{run_id}/{model_name}"
        model_version = mlflow.register_model(model_uri=model_registry_path, name=model_name)
        
        registered_model_version = model_version.version
        registered_model_name = model_version.name
        
        self.logger.info(f"The latest model version in model registry is {registered_model_version}")
        
        self.client.transition_model_version_stage(
            name=registered_model_name,
            version=registered_model_version,
            stage="Staging"
        )
        
        self.logger.info("Model pushed to Staging stage")


    
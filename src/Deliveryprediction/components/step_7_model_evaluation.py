import pandas as pd
import joblib
from Deliveryprediction import logger
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json
from Deliveryprediction.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    
    TARGET_COLUMN = "time_taken"
    def __init__(self,logger, repo_owner, repo_name, experiment_name, config: ModelEvaluationConfig):
        self.config = config
        self.logger = logger
        self.target_column = "time_taken"
        self.root_dir = Path(self.config.root_dir)
        self.root_path = Path(self.config.data_input_dir)
        self.prepro_dir = Path(self.config.prepro_dir)
        self.train_data_path = self.root_path / "train_trans.csv"
        self.test_data_path = self.root_path / "test_trans.csv"
        self.model_path = Path(self.config.model_path)
        self.metric_path = Path(self.config.metric_file)

        
        # self.transformer = None
         
        # Initialize Dagshub and MLflow
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
        mlflow.set_experiment(experiment_name)
    
       
    def load_data(self, data_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(data_path)
            self.logger.info(f"Data loaded successfully from {data_path}")
            return df
        except FileNotFoundError:
            self.logger.error(f"File not found: {data_path}")
            return None
    
    def split_data(self, data: pd.DataFrame):
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        return X, y
    
    def load_model(self, model_path: Path):
        try:
            model = joblib.load(model_path)
            self.logger.info("Model loaded successfully")
            return model
        except FileNotFoundError:
            self.logger.error(f"Model file not found: {model_path}")
            return None
    
    def save_model_info(self, metric_path: Path, run_id, artifact_path, model_name):
        info_dict = {"run_id": run_id, "artifact_path": artifact_path, "model_name": model_name}
        with open(metric_path, "w") as f:
            json.dump(info_dict, f, indent=4)
        self.logger.info("Model information saved")
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
        mean_cv_score = -cv_scores.mean()
        
        self.logger.info("Model evaluation completed")
        return train_mae, test_mae, train_r2, test_r2, mean_cv_score, cv_scores
    
    def log_metrics_to_mlflow(self, model, train_mae, test_mae, train_r2, test_r2, mean_cv_score, cv_scores, X_train, train_data, test_data, root_path):
        with mlflow.start_run() as run:
            mlflow.set_tag("model", "Food Delivery Time Regressor")
            mlflow.log_params(model.get_params())
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("mean_cv_score", mean_cv_score)
            mlflow.log_metrics({f"CV {num}": -score for num, score in enumerate(cv_scores)})
            
            train_data_input = mlflow.data.from_pandas(train_data, targets=self.target_column)
            test_data_input = mlflow.data.from_pandas(test_data, targets=self.target_column)
            mlflow.log_input(dataset=train_data_input, context="training")
            mlflow.log_input(dataset=test_data_input, context="validation")
            
            model_signature = mlflow.models.infer_signature(X_train.sample(20, random_state=42), model.predict(X_train.sample(20, random_state=42)))
            mlflow.sklearn.log_model(model, "delivery_time_pred_model", signature=model_signature)
            
            mlflow.log_artifact(self.root_dir / "models" / "stacking_regressor.joblib")
            mlflow.log_artifact(self.root_dir / "models" / "power_transformer.joblib")
            mlflow.log_artifact(self.prepro_dir)
            
            artifact_uri = mlflow.get_artifact_uri()
            self.logger.info("MLflow logging complete")
            return run.info.run_id, artifact_uri
    
    def run(self):
        
        train_data = self.load_data(self.train_data_path)
        test_data = self.load_data(self.test_data_path)
        X_train, y_train = self.split_data(train_data)
        X_test, y_test = self.split_data(test_data)
        
        model = self.load_model(self.model_path)
        train_mae, test_mae, train_r2, test_r2, mean_cv_score, cv_scores = self.evaluate_model(model, X_train, y_train, X_test, y_test)
        
        run_id, artifact_uri = self.log_metrics_to_mlflow(model, train_mae, test_mae, train_r2, test_r2, mean_cv_score, cv_scores, X_train, train_data, test_data, self.root_path)
        
        save_json_path = self.metric_path
        self.save_model_info(save_json_path, run_id, artifact_uri, "delivery_time_pred_model")

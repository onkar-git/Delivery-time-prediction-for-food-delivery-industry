import pandas as pd
import yaml
import joblib
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor
from Deliveryprediction import logger
from Deliveryprediction.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    TARGET_COLUMN = "time_taken"
    
    def __init__(self, config:ModelTrainerConfig):
        self.config = config
        self.root_path = Path(self.config.data_input_dir)
        self.train_data_path = self.root_path / "train_trans.csv"
        self.test_data_path = self.root_path / "test_trans.csv"
        
        self.save_data_dir = Path(self.config.root_dir)
        self.save_data_dir.mkdir(exist_ok=True, parents=True)

        self.model_save_dir = self.save_data_dir / "models"
        self.model_save_dir.mkdir(exist_ok=True)
        self.training_data = None
        self.model = None
        self.stacking_model = None
        self.transformer = None

    def load_data(self) -> pd.DataFrame:
        try:
            print(self.train_data_path)
            df = pd.read_csv(self.train_data_path)
            logger.info("Training Data read successfully")
            return df
        except FileNotFoundError:
            logger.error("The file to load does not exist")
            return None

    @staticmethod
    def read_params(file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def save_model(model, save_dir: Path, model_name: str):
        joblib.dump(value=model, filename=save_dir / model_name)

    @staticmethod
    def save_transformer(transformer, save_dir: Path, transformer_name: str):
        joblib.dump(transformer, save_dir / transformer_name)

    @staticmethod
    def train_model(model, X_train: pd.DataFrame, y_train):
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def make_X_and_y(data: pd.DataFrame, target_column: str):
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y

    def run(self):
        # Load training data
        self.training_data = self.load_data()
        if self.training_data is None:
            return

        X_train, y_train = self.make_X_and_y(self.training_data, self.TARGET_COLUMN)
        logger.info("Dataset splitting completed")

        
        
        rf = RandomForestRegressor(n_estimators = self.config.n_estimators_rf, 
                                    max_depth = self.config.max_depth_rf,
                                    criterion = self.config.criterion_rf,
                                    max_features =self.config.max_features_rf,
                                    min_samples_split = self.config.min_samples_split_rf,
                                    min_samples_leaf = self.config.min_samples_leaf_rf,
                                    max_samples = self.config.max_samples_rf,
                                    verbose= self.config.verbose_rf,
                                    n_jobs = self.config.n_jobs_rf)


        lgbm = LGBMRegressor(n_estimators = self.config.n_estimators,
                                max_depth = self.config.max_depth,
                                learning_rate = self.config.learning_rate,
                                subsample = self.config.subsample,
                                min_child_weight = self.config.min_child_weight,
                                min_split_gain = self.config.min_split_gain,
                                reg_lambda = self.config.reg_lambda,
                                n_jobs = self.config.n_jobs,)

        lr = LinearRegression()
        power_transform = PowerTransformer()

        stacking_reg = StackingRegressor(estimators=[("rf_model", rf),
                                                     ("lgbm_model", lgbm)],
                                         final_estimator=lr, cv=5, n_jobs=-1)

        self.model = TransformedTargetRegressor(regressor=stacking_reg, transformer=power_transform)
        
        # Train model
        self.train_model(self.model, X_train, y_train)
        logger.info("Model training completed")
        
        self.stacking_model = self.model.regressor_
        self.transformer = self.model.transformer_
        
        # Save models
        self.save_model(self.model, self.model_save_dir, "model.joblib")
        self.save_model(self.stacking_model, self.model_save_dir, "stacking_regressor.joblib")
        self.save_transformer(self.transformer, self.model_save_dir, "power_transformer.joblib")
        logger.info("All models and transformers saved successfully")

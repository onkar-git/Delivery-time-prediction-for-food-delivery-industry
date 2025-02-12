import pandas as pd
import logging
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, 
    MinMaxScaler, 
    OrdinalEncoder)
import joblib
from sklearn import set_config
from Deliveryprediction.entity.config_entity import DataTransformerConfig
from Deliveryprediction import logger

# set the transformer outputs to pandas
set_config(transform_output='pandas')

class DataTransformation:
    def __init__(self, config :DataTransformerConfig):
        self.config = config
        self.root_path = Path(self.config.data_input_dir)
        self.train_data_path = self.root_path / "train.csv"
        self.test_data_path = self.root_path / "test.csv"
        self.save_data_dir = Path(self.config.data_tran_dir)
        self.save_data_dir.mkdir(exist_ok=True, parents=True)
        self.train_trans_filename = "train_trans.csv"
        self.test_trans_filename = "test_trans.csv"
        self.save_train_trans_path = self.save_data_dir / self.train_trans_filename
        self.save_test_trans_path = self.save_data_dir / self.test_trans_filename
        self.transformer_save_dir = self.save_data_dir / "models"
        self.transformer_save_dir.mkdir(exist_ok=True)
        
        self.num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]
        self.nominal_cat_cols = ['weather', 'type_of_order', 'type_of_vehicle', 'festival',
                                 'city_type', 'is_weekend', 'order_time_of_day']
        self.ordinal_cat_cols = ['traffic', 'distance_type']
        self.target_col = 'time_taken'
        self.traffic_order = ["low", "medium", "high", "jam"]
        self.distance_type_order = ["short", "medium", "long", "very_long"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("scale", MinMaxScaler(), self.num_cols),
                ("nominal_encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), self.nominal_cat_cols),
                ("ordinal_encode", OrdinalEncoder(categories=[self.traffic_order, self.distance_type_order],
                                                  encoded_missing_value=-999, handle_unknown="use_encoded_value",
                                                  unknown_value=-1), self.ordinal_cat_cols)
            ],
            remainder="passthrough",
            n_jobs=-1,
            verbose_feature_names_out=False
        )

    def load_data(self, data_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(data_path)
            return df
        except FileNotFoundError:
            logger.error("The file to load does not exist")
            return pd.DataFrame()

    def drop_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"The original dataset has {data.shape[0]} rows and {data.shape[1]} columns")
        df_dropped = data.dropna()
        logger.info(f"The dataset after dropping missing values has {df_dropped.shape[0]} rows and {df_dropped.shape[1]} columns")
        return df_dropped

    def make_X_and_y(self, data: pd.DataFrame):
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]
        return X, y

    def train_preprocessor(self, data: pd.DataFrame):
        self.preprocessor.fit(data)
        return self.preprocessor

    def perform_transformations(self, data: pd.DataFrame):
        return self.preprocessor.transform(data)

    def join_X_and_y(self, X: pd.DataFrame, y: pd.Series):
        return X.join(y, how='inner')

    def save_data(self, data: pd.DataFrame, save_path: Path):
        data.to_csv(save_path, index=False)

    def save_transformer(self):
        joblib.dump(self.preprocessor, self.transformer_save_dir / "preprocessor.joblib")

    def run_transformation_pipeline(self):
        train_df = self.drop_missing_values(self.load_data(self.train_data_path))
        logger.info("Train data loaded successfully")
        test_df = self.drop_missing_values(self.load_data(self.test_data_path))
        logger.info("Test data loaded successfully")
        
        X_train, y_train = self.make_X_and_y(train_df)
        X_test, y_test = self.make_X_and_y(test_df)
        logger.info("Data splitting completed")

        self.train_preprocessor(X_train)
        logger.info("Preprocessor is trained")

        X_train_trans = self.perform_transformations(X_train)
        logger.info("Train data is transformed")
        X_test_trans = self.perform_transformations(X_test)
        logger.info("Test data is transformed")

        train_trans_df = self.join_X_and_y(pd.DataFrame(X_train_trans), y_train)
        test_trans_df = self.join_X_and_y(pd.DataFrame(X_test_trans), y_test)
        logger.info("Datasets joined")

        self.save_data(train_trans_df, self.save_train_trans_path)
        self.save_data(test_trans_df, self.save_test_trans_path)
        logger.info("Transformed data saved")

        self.save_transformer()
        logger.info("Preprocessor saved")


import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from Deliveryprediction.components._3_data_cleaning import DataCleaning


class PredictionPipeline:
    def __init__(self,DataCleaning):
        self.model = joblib.load(Path('artifacts/model_trainer/models/model.joblib'))
        
        
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from Deliveryprediction import logger
from pathlib import Path
import os
from Deliveryprediction.entity.config_entity import DataPreparationConfig



class DataPreparation:

    def __init__(self,config:DataPreparationConfig):
        self.config = config
                
    def load_data(self) -> pd.DataFrame:
        try:
          df = pd.read_csv(self.config.data_input_dir)
        except FileNotFoundError:
            logger.error("The file to load does not exist")
        return df
    def split_data(self, data: pd.DataFrame):
        train_data, test_data = train_test_split(data, 
                                                test_size=self.config.params.test_size, 
                                                random_state=self.config.params.random_state)
        

        train_data.to_csv(os.path.join(self.config.train_dir, "train.csv"),index = False)
        test_data.to_csv(os.path.join(self.config.test_dir, "test.csv"),index = False)

        # return train_data, test_data
  

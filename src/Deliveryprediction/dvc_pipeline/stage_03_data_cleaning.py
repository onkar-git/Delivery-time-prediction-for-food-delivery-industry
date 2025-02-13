from Deliveryprediction.config.configuration import ConfigurationManager
from Deliveryprediction.components._3_data_cleaning import DataCleaning
from Deliveryprediction import logger
from pathlib import Path



STAGE_NAME = "Data Cleaning stage"

class DataCleaningPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("E://projects//Delivery-time-prediction-for-food-devlivery-industry//artifacts//data_validation//status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_cleaning_config = config.get_data_cleaning_config()
                cleaning_data = DataCleaning(config=data_cleaning_config)
                data=cleaning_data.load_data()
                cleaning_data.perform_data_cleaning(data)   
            else:
                raise Exception("You data schema is not valid check artifact of data valiadtion")

        except Exception as e:
            print(e)
import numpy as np
import pandas as pd
from pathlib import Path
from Deliveryprediction import logger
from Deliveryprediction.entity.config_entity import DataCleaningConfig

class DataCleaning:
    def __init__(self, config:  DataCleaningConfig):
         self.config = config

    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.config.data_input_dir)
        return data
    
    columns_to_drop =  ['rider_id',
                    'restaurant_latitude',
                    'restaurant_longitude',
                    'delivery_latitude',
                    'delivery_longitude',
                    'order_date',
                    "order_time_hour",
                    "order_day",
                    "city_name",
                    "order_day_of_week",
                    "order_month"]
    
    def change_column_names(self,data: pd.DataFrame) -> pd.DataFrame:
        return (
        data.rename(str.lower,axis=1)
        .rename({
            "delivery_person_id" : "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"},axis=1)
    )

    def data_cleaning(self,data: pd.DataFrame) -> pd.DataFrame:
        minors_data = data.loc[data['age'].astype('float') < 18]
        minor_index = minors_data.index.tolist()
        six_star_data = data.loc[data['ratings'] == "6"]
        six_star_index = six_star_data.index.tolist()
        return (
            data
            .drop(columns="id")
            .drop(index=minor_index)                                                # Minor riders in data dropped
            .drop(index=six_star_index)                                             # six star rated drivers dropped
            .replace("NaN ",np.nan)                                                 # missing values in the data
            .assign(
                # city column out of rider id
                city_name = lambda x: x['rider_id'].str.split("RES").str.get(0),
                # convert age to float
                age = lambda x: x['age'].astype(float),
                # convert ratings to float
                ratings = lambda x: x['ratings'].astype(float),
                # absolute values for location based columns
                restaurant_latitude = lambda x: x['restaurant_latitude'].abs(),
                restaurant_longitude = lambda x: x['restaurant_longitude'].abs(),
                delivery_latitude = lambda x: x['delivery_latitude'].abs(),
                delivery_longitude = lambda x: x['delivery_longitude'].abs(),
                # order date to datetime and feature extraction
                order_date = lambda x: pd.to_datetime(x['order_date'],
                                                    dayfirst=True),
                order_day = lambda x: x['order_date'].dt.day,
                order_month = lambda x: x['order_date'].dt.month,
                order_day_of_week = lambda x: x['order_date'].dt.day_name().str.lower(),
                is_weekend = lambda x: (x['order_date']
                                        .dt.day_name()
                                        .isin(["Saturday","Sunday"])
                                        .astype(int)),
                # time based columns
                order_time = lambda x: pd.to_datetime(x['order_time'],
                                                    format='mixed'),
                order_picked_time = lambda x: pd.to_datetime(x['order_picked_time'],
                                                            format='mixed'),
                # time taken to pick order
                pickup_time_minutes = lambda x: (
                                                (x['order_picked_time'] - x['order_time'])
                                                .dt.seconds / 60
                                                ),
                # hour in which order was placed
                order_time_hour = lambda x: x['order_time'].dt.hour,
                # time of the day when order was placed
                order_time_of_day = lambda x: (
                                    x['order_time_hour'].pipe(self.time_of_day)),
                # categorical columns
                weather = lambda x: (
                                    x['weather']
                                    .str.replace("conditions ","")
                                    .str.lower()
                                    .replace("nan",np.nan)),
                traffic = lambda x: x["traffic"].str.rstrip().str.lower(),
                type_of_order = lambda x: x['type_of_order'].str.rstrip().str.lower(),
                type_of_vehicle = lambda x: x['type_of_vehicle'].str.rstrip().str.lower(),
                festival = lambda x: x['festival'].str.rstrip().str.lower(),
                city_type = lambda x: x['city_type'].str.rstrip().str.lower(),
                # multiple deliveries column
                multiple_deliveries = lambda x: x['multiple_deliveries'].astype(float),
                # target column modifications
                time_taken = lambda x: (x['time_taken']
                                        .str.replace("(min) ","")
                                        .astype(int)))
            .drop(columns=["order_time","order_picked_time"])
        )
    
    def clean_lat_long(self,data: pd.DataFrame, threshold: float=1.0) -> pd.DataFrame:
        location_columns = ['restaurant_latitude',
                            'restaurant_longitude',
                            'delivery_latitude',
                            'delivery_longitude']

        return (
            data
            .assign(**{
                col: (
                    np.where(data[col] < threshold, np.nan, data[col].values)
                )
                for col in location_columns
            })
        )

    
    # extract day, day name, month and year
    def extract_datetime_features(self,ser: pd.Series) -> pd.DataFrame:
        date_col = pd.to_datetime(ser,dayfirst=True)

        return (
            pd.DataFrame(
                {
                    "day": date_col.dt.day,
                    "month": date_col.dt.month,
                    "year": date_col.dt.year,
                    "day_of_week": date_col.dt.day_name(),
                    "is_weekend": date_col.dt.day_name().isin(["Saturday","Sunday"]).astype(int)
                }
            ))
            

        
    def time_of_day(self, ser: pd.Series):

        return(
            pd.cut(ser,bins=[0,6,12,17,20,24],right=True,
                labels=["after_midnight","morning","afternoon","evening","night"])
        )



    def calculate_haversine_distance(self,data: pd.DataFrame) -> pd.DataFrame:
        location_columns = ['restaurant_latitude',
                            'restaurant_longitude',
                            'delivery_latitude',
                            'delivery_longitude']
        
        lat1 = data[location_columns[0]]
        lon1 = data[location_columns[1]]
        lat2 = data[location_columns[2]]
        lon2 = data[location_columns[3]]

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(
            dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

        c = 2 * np.arcsin(np.sqrt(a))
        distance = 6371 * c

        return ( data.assign(
                distance = distance))


    def create_distance_type(self,data: pd.DataFrame) -> pd.DataFrame:
        return(
            data
            .assign(
                    distance_type = pd.cut(data["distance"],bins=[0,5,10,15,25],
                                            right=False,labels=["short","medium","long","very_long"])
        ))



    def drop_columns(self,data: pd.DataFrame, columns: list) -> pd.DataFrame:
        data=data.drop(columns=columns)
        #cleaned_data= data.dropna()
        return data
    
    def save_data(self,data: pd.DataFrame):
            data = data.dropna()
            #df.to_csv(file_path_pandas, index=False)
            return data.to_csv(self.config.preprocess_data_dir,index=False)
    
    def perform_data_cleaning(self, data: pd.DataFrame):
    
        cleaned_data = (
            data
            .pipe(self.change_column_names)
            .pipe(self.data_cleaning)
            .pipe(self.clean_lat_long)
            .pipe(self.calculate_haversine_distance)
            .pipe(self.create_distance_type)
            .pipe(self.drop_columns,columns=self.columns_to_drop)
            .pipe(self.save_data)
        )
    
        return cleaned_data
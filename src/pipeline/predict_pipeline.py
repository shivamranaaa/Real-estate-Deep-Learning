import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path="C:\\Users\\shiva\\OneDrive\\Desktop\\mlproject-main\\artifacts\\model.pkl"
            # preprocessor_path="C:\\Users\\shiva\\OneDrive\\Desktop\\mlproject-main\\artifacts\\proprocessor.pkl"
            print("Before Loading")
            model=load_object(file_path=model_path)
            # preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            # data_scaled=preprocessor.transform(features)
            preds=model.predict(features)[0]
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        X1_transaction_date:float,
        X2_house_age:float,
        X3_distance_to_the_nearest_MRT_station:float,
        X4_number_of_convenience_stores:float,
        X5_latitude:float,
        X6_longitude:float):

        self.X1_transaction_date = X1_transaction_date

        self.X2_house_age = X2_house_age

        self.X3_distance_to_the_nearest_MRT_station = X3_distance_to_the_nearest_MRT_station

        self.X4_number_of_convenience_stores = X4_number_of_convenience_stores

        self.X5_latitude = X5_latitude

        self.X6_longitude = X6_longitude

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "X1 transaction date": [self.X1_transaction_date],
                "X2 house age": [self.X2_house_age],
                "X3 distance to the nearest MRT station": [self.X3_distance_to_the_nearest_MRT_station],
                "X4 number of convenience stores": [self.X4_number_of_convenience_stores],
                "X5 latitude": [self.X5_latitude],
                "X6 longitude": [self.X6_longitude],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


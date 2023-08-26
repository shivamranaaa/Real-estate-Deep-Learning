import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define CustomException class here
class CustomException(Exception):
    def __init__(self, message, error_detail):
        self.message = message
        self.error_detail = error_detail
        super().__init__(self.message)

from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Perform regression using a neural network with dropout layers
            model = Sequential([
                Dense(128, activation='relu', input_dim=X_train.shape[1]),
                Dropout(0.2),  # Adding dropout with 20% dropout rate
                Dense(64, activation='relu'),
                Dropout(0.2),  # Adding dropout with 20% dropout rate
                Dense(32, activation='relu'),
                Dropout(0.2),  # Adding dropout with 20% dropout rate
                Dense(1, activation='linear')
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=2)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            logging.info(f"Mean Squared Error of the regression model: {mse:.4f}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            return mse
            
        except Exception as e:
            raise CustomException("Error occurred", str(e))

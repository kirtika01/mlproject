import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataInjestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataInjestion:
    def __init__(self):
        self.injestion_config = DataInjestionConfig()

    def initiate_data_injestion(self):
        logging.info("Data Injestion Started")
        try:

            data = pd.read_csv("notebook\data\stud.csv")
            logging.info("Data Injestion Completed")

            os.makedirs(
                os.path.dirname(self.injestion_config.train_data_path), exist_ok=True
            )

            data.to_csv(self.injestion_config.raw_data_path, index=False)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.injestion_config.train_data_path, index=False, header=True
            )

            test_set.to_csv(
                self.injestion_config.test_data_path, index=False, header=True
            )

            logging.info("Train test split completed")
            return (
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path,
            )
        except Exception as e:

            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataInjestion()
    train_data, test_data = obj.initiate_data_injestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path=None))

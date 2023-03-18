import os
import sys
from src.exception_handler import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Data ingestion config class
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'data.csv')
    dataset_path = os.path.join('notebook', 'data', 'stud.csv')

# Data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("Data ingestion config captured")

    def initiate_data_ingestion(self):
        logging.info("Initiated data ingestion process")
        try:
            df = pd.read_csv(self.ingestion_config.dataset_path)
            logging.info("Datasource fetched successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data stored into raw data path")

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data stored into the designated paths")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path    
            )

        except Exception as err:
            # Custom exception raised
            raise CustomException(err, sys)
        
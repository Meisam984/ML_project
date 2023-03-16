import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception_handler import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Numerical and categorical pipelines are mixed into a preprocessor component
    def get_data_transformer_obj(self):
                
        try:
            num_features = ["writing_score", "reading_score"]
            cat_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            logging.info("Split numerical and categorical features")

            num_pipeline = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]
            )
            logging.info("Numerical pipeline created")

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical pipeline created")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )

            return preprocessor
        except Exception as err:
            raise CustomException(err, sys)
            
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test dataframes loaded")

            preprocessor_obj = self.get_data_transformer_obj()
            logging.info("Preprocessor object obtained")

            label = "math_score"

            features_train_df = train_df.drop(label, axis=1)
            label_train_df = train_df[label]

            features_test_df = test_df.drop(label, axis=1)
            label_test_df = test_df[label]
            logging.info("Split both training and test datasets into features and label columns")

            logging.info("Applying preprocessor object onto the train and test features sets initiated")
            features_train_arr = preprocessor_obj.fit_transform(features_train_df)
            features_test_arr = preprocessor_obj.transform(features_test_df)

            train_arr = np.c_[
                features_train_arr, np.array(label_train_df)
            ]

            test_arr = np.c_[
                features_test_arr, np.array(label_test_df)
            ]

            logging.info("Saving preprocessor object into a pickle file")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor_obj
            )
                
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        except Exception as err:
            raise CustomException(err, sys)
               
            
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path, test_path)



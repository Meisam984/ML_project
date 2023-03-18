import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception_handler import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import *

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info(
                "Split training and test data into features and label columns")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regression': KNeighborsRegressor(),
                'XGB Regression': XGBRegressor(),
                'CatBoosting Regression': CatBoostRegressor(verbose=False),
                'AdaBoost Regression': AdaBoostRegressor()
            }

            params = {
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "K-Neighbors Regression": {},
                "XGB Regression": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regression": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regression": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }

            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logging.info("Models scores calculated and returned in a dictionary")
            key_list = list(model_report.keys())
            value_list = list(model_report.values())

            best_model_score = max(sorted(value_list))
            position = value_list.index(best_model_score)
            best_model_name = key_list[position]
            best_model = models[best_model_name]
            logging.info("Best model score captured")

            if best_model_score <= 0.6:
                raise CustomException("There is no acceptable model with score greater than 60%", sys)
            
            save_object(
                file_path=ModelTrainerConfig.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model is selected to be {best_model_name}, with the score of {best_model_score:.2%} saved into 'artifacts\model.pkl'")

            return best_model_score
            

        except Exception as err:
            raise CustomException(err, sys)

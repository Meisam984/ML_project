import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception_handler import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as err:
        raise CustomException(err, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models) -> dict:
    try:
        report = {}
        key_list = list(models.keys())
        value_list = list(models.values())

        for model in value_list:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            position = value_list.index(model)
            report[key_list[position]] = test_model_score
        
        return report
            
    except Exception as err:
        raise CustomException(err, sys)

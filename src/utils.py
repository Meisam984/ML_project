import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception_handler import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as err:
        raise CustomException(err, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, parameters) -> dict:
    try:
        report = {}
        key_list = list(models.keys())
        model_value_list = list(models.values())

        for model in model_value_list:
            position = model_value_list.index(model)
            param = parameters[key_list[position]]
            gs = GridSearchCV(estimator=model, param_grid=param, cv=3)
            
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[key_list[position]] = test_model_score
        
        return report
            
    except Exception as err:
        raise CustomException(err, sys)

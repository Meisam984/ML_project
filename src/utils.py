import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception_handler import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(file_path, file_obj)
    except Exception as err:
        raise CustomException(err, sys)
import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from ssrc.exception import CustomException
from ssrc.logger import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


#save pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Error raised in save_object in utils")
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            #Model Traning
            rs = RandomizedSearchCV(model, para,cv=3)
            rs.fit(X_train,y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train,y_train)

            #make Prediction
            y_test_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info("error raised in evaluate models")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            pickle.load(file_obj)

    except Exception as e:
        logging.info('error raised in load objects')
        raise CustomException(e,sys)

import os
import sys
from ssrc.exception import CustomException
from ssrc.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
from ssrc.component.data_transformation import DataTransformation
from ssrc.component.model_training import ModelTraning
from ssrc.component.data_ingestion import DataIngestion
import warnings
warnings.filterwarnings("ignore")





if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    transform = DataTransformation()
    train_arr,test_arr,_= transform.initiate_data_transformation(train_data,test_data)
    train = ModelTraning()
    train.initatied_model_traning(train_arr,test_arr)
        
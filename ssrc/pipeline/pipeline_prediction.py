import sys
import os
from ssrc.exception import CustomException
from ssrc.logger import logging
from ssrc.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Age:float,
                 Gender:str,
                 Location:str,
                 Subscription_Length_Months:float,
                 Monthly_Bill:float,
                 Total_Usage_GB:float
                 
                 ):
        
        self.Age=Age
        self.Gender=Gender
        self.Location=Location
        self.Subscription_Length_Months=Subscription_Length_Months
        self.Monthly_Bill=Monthly_Bill
        self.Total_usage_GB=Total_Usage_GB


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age':[self.Age],
                'Gender':[self.Gender],
                'Location':[self.Location],
                'Subscription_Monthly_Bill':[self.Subscription_Length_Months],
                'Month_Bill':[self.Monthly_Bill],
                'Total_Usage_GB':[self.Total_usage_GB]

            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
        
if __name__=="__main__":
    predict_obj = PredictPipeline()
    data  = CustomData(22,'Female',17,73.36,236,22)
    df = data.get_data_as_dataframe()
    print(df)
    print(predict_obj.predict(df))
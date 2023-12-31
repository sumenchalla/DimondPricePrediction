import sys,os,pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utlies import load_object


class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            data_scaled=preprocessor.transfrom(features)
            pred=model.predict(data_scaled)

            return pred
        
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e,sys)
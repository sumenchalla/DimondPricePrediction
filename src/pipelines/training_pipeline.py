import os
import sys
from  src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transforamtion import Datatransforamtion
from  src.components.model_trainer import ModelTrainer

if __name__=='__main__':
    obj =DataIngestion()
    logging.info('objecte is created')
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    data_transform=Datatransforamtion()
    train_arr,test_arr,obj_path=data_transform.initiate_data_transfroamtion(train_data_path,test_data_path)

    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)
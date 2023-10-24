import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.exception import CustomException
from src.logger import logging

from src.utlies import save_object
from src.utlies import evaluate_model
from dataclasses import dataclass
import os,sys



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting the dependednt and independent variables from  trainig and test dataset')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                
            )

            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet(),
                "Decisiontree":DecisionTreeRegressor(),
                'Randomforest':RandomForestRegressor()
                    }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n=============================================================================')
            logging.info(f'Model report: {model_report}')

            #to get the best model score from dictionary

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            print(f"Best model is founded,Model Name : {best_model_name} ,R2 scroe : {best_model_score}")
            logging.info(f"Best model is founded,Model Name : {best_model_name} ,R2 scroe : {best_model_score}")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info('Model training has been stoped in the intiation stage')
            raise CustomException(e,sys)
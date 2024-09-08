import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuro_mf import ModelFactory  # Importing ModelFactory from neuro_mf to automate model selection

from us_visa.exception import USvisaException  # Custom exception class for handling errors
from us_visa.logger import logging  # Custom logger for logging messages
from us_visa.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object  # Utility functions for data handling
from us_visa.entity.config_entity import ModelTrainerConfig  # Configuration entity for Model Trainer
from us_visa.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact  # Artifact entities for storing output
from us_visa.entity.estimator import USvisaModel  # Custom model class for US Visa

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        # Initialize the ModelTrainer with data transformation artifacts and model training configuration
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            # Initialize the ModelFactory with the configuration file path
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            # Split the train and test arrays into features (X) and labels (y)
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            # Get the best model based on the training data and base accuracy
            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model  # Retrieve the best model object

            # Predict on the test data using the best model
            y_pred = model_obj.predict(x_test)
            
            # Calculate performance metrics for the best model
            accuracy = accuracy_score(y_test, y_pred) 
            f1 = f1_score(y_test, y_pred)  
            precision = precision_score(y_test, y_pred)  
            recall = recall_score(y_test, y_pred)
            # Create a ClassificationMetricArtifact to store model metrics
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            
            return best_model_detail, metric_artifact  # Return the best model details and metrics
        
        except Exception as e:
            # Handle exceptions and raise custom USvisaException
            raise USvisaException(e, sys) from e
        

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            # Load transformed training and testing data as numpy arrays
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            # Get the best model object and evaluation report using the training and testing data
            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            
            # Load the preprocessing object used during data transformation
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Check if the best model's score is above the expected accuracy threshold
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            # Create a USvisaModel object with both preprocessing and trained model objects
            usvisa_model = USvisaModel(preprocessing_object=preprocessing_obj,
                                       trained_model_object=best_model_detail.best_model)
            logging.info("Created usvisa model object with preprocessor and model")
            logging.info("Created best model file path.")
            
            # Save the trained model object to the specified file path
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)

            # Create a ModelTrainerArtifact to store the trained model file path and metrics
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact  # Return the model trainer artifact
        except Exception as e:
            # Handle exceptions and raise custom USvisaException
            raise USvisaException(e, sys) from e

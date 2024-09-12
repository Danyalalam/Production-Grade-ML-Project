import os
import sys

import numpy as np
import pandas as pd
from us_visa.entity.config_entity import USvisaPredictorConfig  # Importing configuration class for prediction pipeline
from us_visa.entity.s3_estimator import USvisaEstimator  # Importing estimator class for model prediction
from us_visa.exception import USvisaException  # Importing custom exception class
from us_visa.logger import logging  # Importing logging module for logging information
from us_visa.utils.main_utils import read_yaml_file  # Importing utility function to read YAML files
from pandas import DataFrame  # Importing DataFrame from pandas for data manipulation


class USvisaData:
    def __init__(self,
                continent,
                education_of_employee,
                has_job_experience,
                requires_job_training,
                no_of_employees,
                region_of_employment,
                prevailing_wage,
                unit_of_wage,
                full_time_position,
                company_age
                ):
        """
        Constructor for USvisaData class.
        Initializes input data required for prediction based on various features.

        Input Parameters:
        - continent: Continent of the employee
        - education_of_employee: Education level of the employee
        - has_job_experience: Whether the employee has job experience or not
        - requires_job_training: Whether the job requires training
        - no_of_employees: Number of employees in the company
        - region_of_employment: Region where the employee is employed
        - prevailing_wage: Wage paid for the job
        - unit_of_wage: Unit of the wage (e.g., hourly, weekly)
        - full_time_position: Whether the position is full-time
        - company_age: Age of the company
        """
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age

        except Exception as e:
            # Raise a custom exception in case of an error
            raise USvisaException(e, sys) from e

    def get_usvisa_input_data_frame(self) -> DataFrame:
        """
        Converts the input data into a pandas DataFrame for model prediction.

        Returns:
        - DataFrame: Input data formatted as a pandas DataFrame.
        """
        try:
            # Convert input data to a dictionary format
            usvisa_input_dict = self.get_usvisa_data_as_dict()
            # Return DataFrame created from the dictionary
            return DataFrame(usvisa_input_dict)
        
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_usvisa_data_as_dict(self):
        """
        Converts the input data into a dictionary format for easier manipulation.

        Returns:
        - dict: Input data formatted as a dictionary.
        """
        logging.info("Entered get_usvisa_data_as_dict method of USvisaData class")

        try:
            # Create a dictionary with input data
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logging.info("Created usvisa data dict")
            logging.info("Exited get_usvisa_data_as_dict method of USvisaData class")

            # Return the created dictionary
            return input_data

        except Exception as e:
            raise USvisaException(e, sys) from e


class USvisaClassifier:
    def __init__(self, prediction_pipeline_config: USvisaPredictorConfig = USvisaPredictorConfig()) -> None:
        """
        Constructor for USvisaClassifier class.
        Initializes the prediction pipeline configuration.

        Input Parameters:
        - prediction_pipeline_config: Configuration for the prediction pipeline
        """
        try:
            # Initialize prediction pipeline configuration
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise USvisaException(e, sys)

    def predict(self, dataframe) -> str:
        """
        Predicts the output using the trained model.

        Input Parameters:
        - dataframe: DataFrame containing input data for prediction

        Returns:
        - str: Prediction result in string format
        """
        try:
            logging.info("Entered predict method of USvisaClassifier class")
            # Load the model from S3 using the configuration
            model = USvisaEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            # Predict using the model and return the result
            result = model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise USvisaException(e, sys)

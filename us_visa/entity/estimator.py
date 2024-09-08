import sys  # Importing sys module for system-specific parameters and functions
from pandas import DataFrame  # Importing DataFrame from pandas for handling data
from sklearn.pipeline import Pipeline  # Importing Pipeline from sklearn for preprocessing

from us_visa.exception import USvisaException  # Importing custom exception class for handling errors
from us_visa.logger import logging  # Importing logging module for logging

# Class for mapping target values to integers and vice versa
class TargetValueMapping:
    def __init__(self):
        # Mapping Certified to 0 and Denied to 1
        self.Certified: int = 0
        self.Denied: int = 1

    # Method to return the dictionary representation of the class
    def _asdict(self):
        return self.__dict__

    # Method to reverse the mapping of values to their keys
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

# Class to handle the machine learning model and preprocessing for predictions
class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object  # Object for preprocessing
        self.trained_model_object = trained_model_object  # Object for trained model

    # Method to make predictions using the preprocessing object and trained model
    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        logging.info("Entered predict method of UTruckModel class")

        try:
            logging.info("Using the trained model to get predictions")
            # Transform the input dataframe using the preprocessing object
            transformed_feature = self.preprocessing_object.transform(dataframe)
            logging.info("Used the trained model to get predictions")
            # Use the trained model to predict the transformed features
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            # Raise custom exception if there is an error during prediction
            raise USvisaException(e, sys) from e

    # Representation method to return the type of trained model object
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    # String method to return the type of trained model object
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

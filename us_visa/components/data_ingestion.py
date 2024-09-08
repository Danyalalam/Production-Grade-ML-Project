import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

# Import necessary entities and utilities
from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.data_access.usvisa_data import USvisaData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initializes the DataIngestion class with the provided configuration for data ingestion.
        
        :param data_ingestion_config: Configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise USvisaException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Exports data from MongoDB into a CSV file and saves it in the feature store.

        Output:
            - Returns a DataFrame containing data exported from MongoDB.
        
        On Failure:
            - Raises an exception if any error occurs during data export.
        """
        try:
            logging.info(f"Exporting data from mongodb")
            
            # Create an instance of USvisaData to interact with MongoDB
            usvisa_data = USvisaData()
            
            # Export the collection as a DataFrame
            dataframe = usvisa_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            
            # Get the feature store file path from the configuration
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            
            # Create the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            
            # Save the DataFrame as a CSV file in the feature store
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            
            return dataframe

        except Exception as e:
            raise USvisaException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Splits the DataFrame into training and testing sets and saves them as CSV files.
        
        Output:
            - Creates train and test CSV files in the specified paths.
        
        On Failure:
            - Raises an exception if any error occurs during data splitting.
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            # Split the data into training and testing sets
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")
            
            # Get the directory path for the training and testing files
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            
            # Save the training and testing sets as CSV files
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the data ingestion process by calling the necessary methods.
        
        Output:
            - Returns a DataIngestionArtifact containing paths to train and test datasets.
        
        On Failure:
            - Raises an exception if any error occurs during data ingestion.
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            # Export data from MongoDB to the feature store
            dataframe = self.export_data_into_feature_store()

            logging.info("Got the data from mongodb")

            # Split the exported data into training and testing sets
            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")

            # Create DataIngestionArtifact object with paths to the training and testing files
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e

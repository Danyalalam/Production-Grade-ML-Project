import sys

# Import necessary libraries
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

# Import custom modules and entities
from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa.entity.estimator import TargetValueMapping

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        Initializes the DataTransformation class with artifacts from data ingestion and validation stages,
        as well as configuration for data transformation.

        :param data_ingestion_artifact: Contains paths to ingested data
        :param data_transformation_config: Configuration settings for data transformation
        :param data_validation_artifact: Contains validation status and messages
        """
        try:
            # Assign provided artifacts and configuration to instance variables
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            
            # Read schema configuration file
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads data from a CSV file.

        :param file_path: Path to the CSV file
        :return: DataFrame containing the data from the file
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)
    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object that can preprocess the data using various techniques
        such as scaling, encoding, and power transformations.

        :return: A preprocessor object created using ColumnTransformer
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize different transformers for various types of data
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            # Fetch columns for each transformation from the schema config file
            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            logging.info("Initialize PowerTransformer")

            # Pipeline to apply power transformation to specified columns
            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            
            # ColumnTransformer to apply transformations on different columns
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process, applying all preprocessing steps to both training and testing datasets.

        :return: DataTransformationArtifact containing paths to the transformed data files and preprocessor object
        """
        try:
            # Check if the validation status is true before proceeding
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                
                # Get the preprocessor object
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                # Read training and testing data
                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                # Separate input features and target variable for training data
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got train features and test features of Training dataset")

                # Calculate company age and add it as a new feature
                input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
                logging.info("Added company_age column to the Training dataset")

                # Drop unnecessary columns
                drop_cols = self._schema_config['drop_columns']
                logging.info("drop the columns in drop_cols of Training dataset")

                input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)
                
                # Replace target values with their mapped encodings
                target_feature_train_df = target_feature_train_df.replace(TargetValueMapping()._asdict())

                # Separate input features and target variable for testing data
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                # Calculate company age and add it as a new feature for test data
                input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']
                logging.info("Added company_age column to the Test dataset")

                # Drop unnecessary columns for test data
                input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)
                logging.info("drop the columns in drop_cols of Test dataset")

                # Replace target values with their mapped encodings for test data
                target_feature_test_df = target_feature_test_df.replace(TargetValueMapping()._asdict())
                logging.info("Got train features and test features of Testing dataset")

                # Apply preprocessing to both training and testing data
                logging.info("Applying preprocessing object on training dataframe and testing dataframe")
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info("Used the preprocessor object to fit transform the train features")

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info("Used the preprocessor object to transform the test features")

                # Apply SMOTEENN to handle imbalanced data
                logging.info("Applying SMOTEENN on Training dataset")
                smt = SMOTEENN(sampling_strategy="minority")
                input_feature_train_final, target_feature_train_final = smt.fit_resample(input_feature_train_arr, target_feature_train_df)
                logging.info("Applied SMOTEENN on training dataset")

                logging.info("Applying SMOTEENN on testing dataset")
                input_feature_test_final, target_feature_test_final = smt.fit_resample(input_feature_test_arr, target_feature_test_df)
                logging.info("Applied SMOTEENN on testing dataset")

                # Combine input features and target variable back into arrays
                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
                test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

                # Save the preprocessor object and transformed data
                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")
                logging.info("Exited initiate_data_transformation method of Data_Transformation class")

                # Create and return DataTransformationArtifact containing paths to transformed data and objects
                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                # If validation fails, raise an exception with the message from the validation artifact
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise USvisaException(e, sys) from e

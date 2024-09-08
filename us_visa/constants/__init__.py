import os
from datetime import date

# Database and Collection names for MongoDB
DATABASE_NAME = "US_VISA"
COLLECTION_NAME = "visa_data"

# Environment variable key for MongoDB URL
MONGODB_URL_KEY = "MONGODB_URL"

# Name of the ML pipeline
PIPELINE_NAME = "usvisa"

# Directory to store artifacts like trained models
ARTIFACT_DIR = "artifact"

# Filename for storing the trained model
MODEL_FILE_NAME = "model.pkl"

# Filenames for data storage
FILE_NAME: str = "usvisa.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Target column for prediction
TARGET_COLUMN = "case_status"

# Current year using Python's datetime module
CURRENT_YEAR = date.today().year

# Filename for storing preprocessing objects
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing_object.pkl"

# Path to the schema configuration file (YAML format)
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

'''
Data Ingestion Constants (Start with DATA_INGESTION_)
'''

# Collection name specifically for data ingestion
DATA_INGESTION_COLLECION_NAME = "visa_data"

# Directory for data ingestion artifacts
DATA_INGESTION_DIR_NAME = "data_ingestion"

# Directory to store feature-extracted data during ingestion
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# Directory for ingested data files
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Ratio for splitting data into training and testing datasets
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation Constants (Start with DATA_VALIDATION_)
"""

# Directory for data validation artifacts
DATA_VALIDATION_DIR_NAME: str = "data_validation"

# Directory to store data drift reports
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

# Filename for the data drift report (YAML format)
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation Constants (Start with DATA_TRANSFORMATION_)
"""

# Directory for data transformation artifacts
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

# Directory to store transformed data
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"

# Directory to store objects used during data transformation (e.g., encoders)
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
Model Trainer Constants (Start with MODEL_TRAINER_)
"""

# Directory for model training artifacts
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

# Directory to store the trained model
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

# Filename for the trained model
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

# Expected minimum score for the model to be considered acceptable
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

# Path to the model configuration file (YAML format)
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

"""
MODEL EVALUATION related constant 
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "usvisa-model2024"
MODEL_PUSHER_S3_KEY = "model-registry"

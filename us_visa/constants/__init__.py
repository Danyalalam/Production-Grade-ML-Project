import os
from datetime import date

DATABASE_NAME = "US_VISA"
COLLECTION_NAME = "visa_data"
MONGODB_URL_KEY = "MONGODB_URL"
PIPELINE_NAME = "usvisa"
ARTIFACT_DIR="artifact"
MODEL_FILE_NAME="model.pkl"
FILE_NAME: str = "usvisa.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
PREPRPOCESSING_OBJECT_FILE_NAME = "preprocessing_object.pkl"
SCHEMA_FILE_PATH=os.path.join("config","schema.yaml")

'''
Data Igestion Constant start with DATA_INGESTION VAR NAME

'''

DATA_INGESTION_COLLECION_NAME = "visa_data"
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
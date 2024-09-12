import os  # Importing os for file and directory path handling
from us_visa.constants import *  # Importing all constants
from dataclasses import dataclass  # Importing dataclass decorator
from datetime import datetime  # Importing datetime module to get the current timestamp

# Generating a timestamp to uniquely identify the pipeline run
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data class to store training pipeline configurations like pipeline name and artifact directory
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME  # Name of the pipeline
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)  # Directory to store all artifacts
    timestamp: str = TIMESTAMP  # Timestamp for the current run

# Creating a global object of the training pipeline configuration class
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

# Data class to store configurations for data ingestion
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)  # Directory to store data ingestion artifacts
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)  # Path to feature store file
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)  # Path to training data file
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)  # Path to testing data file
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO  # Ratio for train-test split
    collection_name: str = DATA_INGESTION_COLLECION_NAME  # Collection name in the database

# Data class to store configurations for data validation
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)  # Directory to store data validation artifacts
    drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)  # Path to the drift report file

# Data class to store configurations for data transformation
@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)  # Directory to store data transformation artifacts
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRAIN_FILE_NAME.replace("csv", "npy"))  # Path to transformed training data file
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TEST_FILE_NAME.replace("csv", "npy"))  # Path to transformed test data file
    transformed_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCESSING_OBJECT_FILE_NAME)  # Path to preprocessed object file

# Data class to store configurations for model training
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)  # Directory to store model trainer artifacts
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)  # Path to trained model file
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE  # Expected accuracy of the model
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH  # Path to model configuration file

@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME
    
@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME
    
@dataclass
class USvisaPredictorConfig:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME
    
    
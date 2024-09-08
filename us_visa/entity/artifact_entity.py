from dataclasses import dataclass  # Importing dataclass decorator from the dataclasses module

# Data class to store paths for training and test data generated during data ingestion
@dataclass
class DataIngestionArtifact:
    trained_file_path: str  # Path to the training data file
    test_file_path: str  # Path to the test data file
    
# Data class to store the results of data validation including drift report
@dataclass
class DataValidationArtifact:
    validation_status: bool  # Whether data validation passed or failed
    message: str  # Message regarding validation status
    drift_report_file_path: str  # Path to the data drift report file
    
# Data class to store file paths for transformed data and objects
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str  # Path to the preprocessed object file
    transformed_train_file_path: str  # Path to the transformed training data file
    transformed_test_file_path: str  # Path to the transformed test data file
    
# Data class to store classification metrics such as f1 score, precision, and recall
@dataclass
class ClassificationMetricArtifact:
    f1_score: float  # F1 score of the model
    precision_score: float  # Precision score of the model
    recall_score: float  # Recall score of the model

# Data class to store the trained model file path and its classification metrics
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str  # Path to the trained model file
    metric_artifact: ClassificationMetricArtifact  # Classification metrics of the model

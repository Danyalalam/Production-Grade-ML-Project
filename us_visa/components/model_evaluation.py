from us_visa.entity.config_entity import ModelEvaluationConfig  # Import model evaluation configuration entity
from us_visa.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact  # Import artifact entities for different pipeline stages
from sklearn.metrics import f1_score  # Import f1_score metric for model evaluation
from us_visa.exception import USvisaException  # Custom exception class for handling errors
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR  # Import constants
from us_visa.logger import logging  # Custom logger for logging messages
import sys  # Import sys for system-specific parameters and functions
import pandas as pd  # Import pandas for data manipulation
from typing import Optional  # Import Optional for optional type hinting
from us_visa.entity.s3_estimator import USvisaEstimator  # Import estimator class for managing models in S3
from dataclasses import dataclass  # Import dataclass decorator for defining simple classes
from us_visa.entity.estimator import USvisaModel  # Import model class
from us_visa.entity.estimator import TargetValueMapping  # Import target value mapping class

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float  # F1 score of the trained model
    best_model_f1_score: float  # F1 score of the best model from production
    is_model_accepted: bool  # Boolean indicating if the trained model is accepted
    difference: float  # Difference between the F1 scores of the trained model and the best model

class ModelEvaluation:
    """
    This class handles the evaluation of the newly trained model against the existing production model.
    """

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        Initialize the ModelEvaluation class with configuration and artifacts.
        """
        try:
            self.model_eval_config = model_eval_config  # Model evaluation configuration
            self.data_ingestion_artifact = data_ingestion_artifact  # Data ingestion artifact
            self.model_trainer_artifact = model_trainer_artifact  # Model trainer artifact
        except Exception as e:
            raise USvisaException(e, sys) from e  # Handle exceptions and raise custom exception

    def get_best_model(self) -> Optional[USvisaEstimator]:
        """
        Description :   This function is used to get the model currently in production from S3 storage
        Method Name :   get_best_model
        
        Output      :   Returns model object if available in S3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            # Get S3 bucket name and model path from configuration
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            
            # Initialize the estimator for fetching the model from S3
            usvisa_estimator = USvisaEstimator(bucket_name=bucket_name, model_path=model_path)

            # Check if the model is present in S3 storage
            if usvisa_estimator.is_model_present(model_path=model_path):
                return usvisa_estimator  # Return the best model if available
            return None  # Return None if no model is present
        except Exception as e:
            raise USvisaException(e, sys)  # Handle exceptions and raise custom exception

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate the newly trained model against the production model
        
        Output      :   Returns EvaluateModelResponse object based on the evaluation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            # Load the test dataset from the data ingestion artifact
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # Add a new feature 'company_age' calculated from the year of establishment
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']

            # Separate features (X) and target (y)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict())  # Replace target values with mapped values

            # Get F1 score of the trained model
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score = None  # Initialize best model F1 score
            best_model = self.get_best_model()  # Get the best model from S3

            # Evaluate the best model's performance if it exists
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)  # Predict using the best model
                best_model_f1_score = f1_score(y, y_hat_best_model)  # Calculate F1 score

            # Calculate difference and check if the trained model is better
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score)
            
            logging.info(f"Result: {result}")  # Log the evaluation result
            return result  # Return the evaluation response

        except Exception as e:
            raise USvisaException(e, sys)  # Handle exceptions and raise custom exception

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function initiates all steps of the model evaluation
        
        Output      :   Returns ModelEvaluationArtifact object
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            # Evaluate the trained model and compare it with the production model
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path  # Get the model path from S3

            # Create a ModelEvaluationArtifact based on evaluation results
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")  # Log the evaluation artifact
            return model_evaluation_artifact  # Return the model evaluation artifact
        except Exception as e:
            raise USvisaException(e, sys) from e  # Handle exceptions and raise custom exception

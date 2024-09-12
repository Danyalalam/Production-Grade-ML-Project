# US Visa Approval Prediction

This project aims to build a machine learning pipeline for predicting US visa approvals using historical data. The end-to-end solution includes data ingestion, transformation, model training, evaluation, and deployment using cloud-based CI/CD with AWS and Docker.

![Project Image](link-to-image-if-any)

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Project Architecture](#project-architecture)
4. [Setup and Installation](#setup-and-installation)
5. [Data Pipeline Components](#data-pipeline-components)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Deployment](#deployment)
8. [Project Structure](#project-structure)

## Introduction

The goal of this project is to automate the process of predicting whether a US visa application will be approved or denied based on various input features. This solution involves building a robust machine learning model, integrating with cloud services, and deploying it using Docker and AWS.



## Features

- **End-to-End Machine Learning Pipeline**: From data ingestion to model deployment.
- **Cloud-Based Deployment**: CI/CD setup using GitHub Actions and AWS services.
- **Dockerized Application**: For consistent deployment and scalability.
- **Data Validation and Transformation**: Ensures data quality before training.
- **Model Evaluation and Selection**: Compares new models against the best model in production.
- **Prediction API**: A RESTful API to get predictions for new data points.

## Project Architecture

The project is organized into various modules, each responsible for a specific step in the pipeline:

1. **Data Ingestion**: Reads and stores raw data.
2. **Data Validation**: Validates schema and data integrity.
3. **Data Transformation**: Prepares the data for model training.
4. **Model Training**: Trains the machine learning model and stores it.
5. **Model Evaluation**: Compares trained models with the best model in production.
6. **Model Pusher**: Deploys the best model to AWS for inference.
7. **Prediction Pipeline**: Provides an interface for new data to be processed and predicted.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Docker
- Git
- AWS CLI configured with necessary permissions
- GitHub account for CI/CD

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/us-visa-approval-prediction.git
    cd us-visa-approval-prediction
    ```

2. **Create a virtual environment and activate it**:
    ```bash
    python -m venv visaa
    source visaa/bin/activate  # For Linux/Mac
    visaa\Scripts\activate  # For Windows
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    Configure the `.env` file with necessary AWS credentials and configurations.

## Data Pipeline Components

### 1. Data Ingestion
- Reads the raw data from a source and stores it in a structured format.
- File: `data_ingestion.py`

### 2. Data Validation
- Checks the schema and ensures data integrity.
- File: `data_validation.py`

### 3. Data Transformation
- Prepares the data for model training by handling missing values, encoding categorical variables, etc.
- File: `data_transformation.py`

## Model Training and Evaluation

### Model Training
- Trains the machine learning model using algorithms like Random Forest, Decision Trees, etc.
- File: `model_trainer.py`

### Model Evaluation
- Compares the new model with the currently deployed model using F1 Score and other metrics.
- File: `model_evaluation.py`

### Model Pusher
- Pushes the best model to the AWS S3 bucket and deploys it on EC2 instances.
- File: `model_pusher.py`

## Deployment

### Docker Setup

- **Dockerfile**: Contains the instructions to build a Docker image of the application.
- **AWS Elastic Container Registry (ECR)**: Stores Docker images.
- **AWS EC2**: Hosts the Docker container to run the application.
- **AWS S3**: Stores the trained models and data artifacts.

### CI/CD Pipeline

- **GitHub Actions Workflow (`aws.yaml`)**:
  - Builds the Docker image.
  - Pushes the Docker image to ECR.
  - Deploys the container on an EC2 instance.
  - File: `.github/workflows/aws.yaml`

## Project Structure

The project is organized into several directories, each serving a specific purpose:

```plaintext
US-VISA-APPROVAL-PREDICTION/
├── .github/
│   └── workflows/
│       └── aws.yaml              # CI/CD workflow for AWS deployment
├── artifact/                     # Directory to store model artifacts
├── config/                       # Configuration files for the project
│   ├── model.yaml                # Model-related configuration
│   └── schema.yaml               # Schema validation configuration
├── logs/                         # Directory to store log files
├── notebook/                     # Jupyter notebooks for data exploration and experimentation
├── static/                       # Static files for the web interface (if any)
├── templates/                    # HTML templates for the web interface (if any)
├── us_visa/                      # Main package for the US Visa prediction project
│   ├── cloud_storage/            # Module for managing cloud storage (e.g., AWS S3)
│   ├── components/               # Directory containing core components for data processing
│   │   ├── data_ingestion.py     # Data ingestion script
│   │   ├── data_transformation.py # Data transformation script
│   │   ├── data_validation.py    # Data validation script
│   │   ├── model_evaluation.py   # Model evaluation script
│   │   ├── model_pusher.py       # Model deployment script
│   │   └── model_trainer.py      # Model training script
│   ├── configuration/            # Configuration management module
│   ├── constants/                # Module for defining constants used in the project
│   ├── data_access/              # Module for data access utilities
│   ├── entity/                   # Data classes and entities
│   ├── exception/                # Custom exception handling module
│   ├── logger/                   # Logging configuration and utilities
│   ├── pipeline/                 # Pipeline orchestration scripts
│   └── utils/                    # Utility functions and helpers
├── us_visa.egg-info/             # Package metadata for the Python project
├── .bashrc                       # Shell script for environment configuration
├── .dockerignore                 # Files and directories to ignore in Docker builds
├── .gitignore                    # Files and directories to ignore in Git
├── app.py                        # Main Flask application file (if any)
├── demo.py                       # Demo script to test the pipeline (if any)
├── Dockerfile                    # Docker configuration file for containerization
└── README.md                     # Project documentation (this file)



### Example Request
```bash
curl -X POST http://<ec2-instance-ip>:<port>/predict -H "Content-Type: application/json" -d '{
  "continent": "Asia",
  "education_of_employee": "Bachelors",
  "has_job_experience": "Yes",
  "requires_job_training": "No",
  "no_of_employees": 200,
  "region_of_employment": "Northeast",
  "prevailing_wage": 85000,
  "unit_of_wage": "Year",
  "full_time_position": "Y",
  "company_age": 15
}'



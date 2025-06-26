import logging
import mlflow
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin # Importing RegressorMixin for type hinting
# from steps.config import ModelNameConfig  # Importing ModelNameConfig for model configuration

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker # Initialize MLflow experiment tracking



@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,  
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    # config: ModelNameConfig,
    ) -> RegressorMixin:
    """
    Trains a linear regression model using the provided training data.
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
    Returns:
        RegressorMixin: The trained linear regression model.
    """
    try:
        # model = None
        # if config.model_name == "LinearRegression":
        mlflow.sklearn.autolog()  # Enable MLflow autologging for scikit-learn models
        model = LinearRegressionModel()
        trained_model = model.train(X_train, y_train)
        return trained_model
        # else:
            # raise ValueError(f"Model {config.model_name} is not supported.")
            # logging.error(f"Model {config.model_name} is not supported.")
    except Exception as e:
        logging.error(f"Error in train_model step: {e}")
        raise e 
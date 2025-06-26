import logging
from zenml import step
import pandas as pd
import mlflow
from src.eval import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated  # For type hinting with ZenML steps
from zenml.client import Client 



experiment_tracker = Client().active_stack.experiment_tracker  # Initialize MLflow experiment tracking

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model( 
     model: RegressorMixin,
     X_test: pd.DataFrame,
     y_test: pd.DataFrame
) -> Tuple[
     Annotated[float,"r2_score"],
     Annotated[float,"rmse"]
]:
    """
     Evaluates the trained model using various metrics such as MSE, R2, and RMSE.         
    Args:
        model (RegressorMixin): The trained regression model to evaluate.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.
    Returns:
        None
    """
    try:
        predictions = model.predict(X_test)
        mse_class= MSE()
        mse= mse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("mse", mse)
        r2_class = R2()
        r2_score = r2_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("r2_score", r2_score)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("rmse", rmse)
        
        return r2_score, rmse

    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise e


    pass
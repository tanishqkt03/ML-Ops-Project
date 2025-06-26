import logging
from abc import ABC, abstractmethod
import numpy as np  
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract class defining the interface for model evaluation.
    """ 
    @abstractmethod
    def calculate_scores(self, model, y_true: np.ndarray, y_pred: np.ndarray):
        """"
        Calculate evaluation scores for the model.
        Args:
            model: The trained model to evaluate.
            y_true: True labels.
            y_pred: Predicted labels.
        """
        pass
    
class MSE(Evaluation):
    """
    Evaluation strategy for calculating Mean Squared Error (MSE).
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate Mean Squared Error (MSE) for the model.
        
        Args:
            model: The trained model to evaluate.
            y_true: True labels.
            y_pred: Predicted labels.
        
        Returns:
            float: The calculated MSE score.
        """
        try:
            logging.info("Calculating Mean Squared Error (MSE)...")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e
        
class R2(Evaluation):
    """
    Evaluation strategy for calculating R-squared (R2) score.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate R-squared (R2) score for the model.
        
        Args:
            model: The trained model to evaluate.
            y_true: True labels.
            y_pred: Predicted labels.
        
        Returns:
            float: The calculated R2 score.
        """
        try:
            logging.info("Calculating R-squared (R2) score...")
            
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R-squared (R2) score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 score: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation strategy for calculating Root Mean Squared Error (RMSE).
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate Root Mean Squared Error (RMSE) for the model.
        
        Args:
            model: The trained model to evaluate.
            y_true: True labels.
            y_pred: Predicted labels.
        
        Returns:
            float: The calculated RMSE score.
        """
        try:
            logging.info("Calculating Root Mean Squared Error (RMSE)...")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e
import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class defining the interface for model operations.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model with training data.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Concrete implementation of a linear regression model.
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Train the linear regression model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
        Returns:
            None
        """
        try:   
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("Linear Regression model trained successfully.")
            return reg
        except Exception as e:
            logging.error(f"Error in training Linear Regression model: {e}")
            raise e
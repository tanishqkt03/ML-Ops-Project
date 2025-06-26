import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStratergy(ABC):
    """
    Abstract class defining strategy for handelling data
    """
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass
    
class DataPreProcessStrategy(DataStratergy):
    """
    strategy for handelling data
    """
    
    def handle_data(self, data:pd.DataFrame)-> pd.DataFrame:
        """
        Preprocess Data
        """
        try:
            data= data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
                axis=1
            )
            data["product_weight_g"].fillna(data["product_weight_g"].mean(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].mean(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].mean(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].mean(), inplace=True)
            data["review_comment_message"].fillna("No Review", inplace=True)
            
            data= data.select_dtypes(include=[np.number])
            data = data.drop(["customer_zip_code_prefix","order_item_id"], axis=1)
            return data
        except Exception as e:
            logging.error(f"Error in DataPreProcessStrategy: {e}")
            raise e 
        
class DataDivideStrategy(DataStratergy):
    """
    Strategy for dividing data into training and testing sets
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide Data
        """
        try:
            X = data.drop(["review_score"], axis=1) #target is review_score
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in DataDivideStrategy: {e}")
            raise e
        
class DataCleaning:
    """
    Class to handle data cleaning and preprocessing /divide into train test
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStratergy):
        self.data = data    
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Clean Data using the provided strategy
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in DataCleaning: {e}")
            raise e
   
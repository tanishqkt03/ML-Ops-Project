import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreProcessStrategy, DataDivideStrategy,DataCleaning
from typing_extensions import Annotated # Annotated is used for type hinting in ZenML steps
from typing import Tuple

@step
def clean_df(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    clarans the data by removing unnecessary columns, filling missing values,
    and dividing the data into training and testing sets.           
    
    Args:
        df (pd.DataFrame): The input DataFrame containing raw data. 
    Returns:
        X_Train (pd.DataFrame): Training features.
        X_Test (pd.DataFrame): Testing features. 
        y_Train (pd.Series): Training labels.
        y_Test (pd.Series): Testing labels.
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()  
        logging.info("Data cleaning and preprocessing completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in clean_df step: {e}")
        raise e
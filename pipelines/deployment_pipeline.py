import json
import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from typing import cast
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from pipelines.utils import get_data_for_test
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings = DockerSettings(
    required_integrations={MLFLOW}
)

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Dynamically import and return test data as a JSON string."""
    data = get_data_for_test()
    return data

@step
def deployment_trigger(accuracy: float) -> bool:
    min_accuracy = 0.9
    if accuracy >= min_accuracy:
        print(f"✅ Accuracy {accuracy} >= {min_accuracy} → Will deploy")
        return True
    else:
        print(f"❌ Accuracy {accuracy} < {min_accuracy} → Skip deployment")
        return False


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    # running: bool = False,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """
    Loads the MLflow model deployment service.
    """
    mlflow_model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    
    # This returns a list of services
    existing_services = mlflow_model_deployer.find_model_server(
    pipeline_name=pipeline_name,
    pipeline_step_name=pipeline_step_name,
    model_name=model_name,
    # running=False,  # allows loading even if paused
)


    if not existing_services:
        raise RuntimeError(
            f"No running service found for pipeline='{pipeline_name}', "
            f"step='{pipeline_step_name}', model='{model_name}'"
        )

    # Cast the first service in the list to the correct type
    service = cast(MLFlowDeploymentService, existing_services[0])
    return service

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run inference request against a prediction service."""
    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns", None)
    data.pop("index", None)

    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_length",
        "product_description_length",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    prediction_input = np.array(json_list)

    prediction = service.predict(prediction_input)
    return prediction

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continous_deployment_pipeline(
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path="data\\olist_customers_dataset.csv")
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    deploy_flag = deployment_trigger(accuracy=rmse)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy_flag,
        workers=workers,
        timeout=timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
    )
    predictor(service=model_deployment_service, data=batch_data)

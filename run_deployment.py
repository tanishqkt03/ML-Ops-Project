from typing import cast

import click
from pipelines.deployment_pipeline import continous_deployment_pipeline, inference_pipeline
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from typing import cast
from zenml.integrations.mlflow.services import MLFlowDeploymentService



DEPLOY="deploy"
PREDICT="predict"
DEPLOY_AND_PREDICT="deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "--c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Choose the deployment configuration: deploy, predict, or deploy_and_predict.",
)

@click.option(
    "--blocking",
    is_flag=True,
    default=False,
    help="Run the deployment pipeline in blocking mode (i.e., keep the model server running in the foreground)."
)


@click.option(
    "--min-accuracy",
    default=0.9,
    help="Minimum accuracy threshold for deploying the model.",
)

def main(config:str, min_accuracy:float, blocking:bool = True):
    """
    Run the deployment pipeline based on the specified configuration.
    """
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config in [DEPLOY, DEPLOY_AND_PREDICT]
    predict = config in [PREDICT, DEPLOY_AND_PREDICT]
    
    if deploy:
        print(f"Running deployment pipeline with min accuracy {min_accuracy}...")
        # Define docker_settings as needed for your environment, or remove if not required
        docker_settings = {}
        continous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60,
            # settings={"docker": docker_settings, "blocking": blocking}
        )

        print("Deployment pipeline completed.")
    if predict:
        print("Running inference pipeline...")
        inference_pipeline(
            pipeline_name="continous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",    
        )
        print("Inference pipeline completed.") 
    
    print(
        "You can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}"
        "[/italic green]\n ...to inspect your experiment runs within the MLflow"
        " UI.\nYou can find your runs tracked within the "
        "`mlflow_example_pipeline` experiment. There you'll also be able to "
        "compare two or more runs.\n\n"
    )
    
     # fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as a daemon "
                f"process service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )


if __name__ == "__main__":
    main()
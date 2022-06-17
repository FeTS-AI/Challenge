"""MLCube handler file"""
from pathlib import Path
import typer
import yaml
from src.my_logic import run_inference


# This is used to create a simple CLI``
app = typer.Typer()


class InferTask(object):
    """ Inference task
    This class defines the environment variables:
        data_path: Directory path to dataset
        output_path: Directory path to final results
        checkpoint_path: Directory path to model checkpoints
        All other parameters are defined in parameters_file
    The `run` method executes the run_inference method from the src.my_logic module"""

    @staticmethod
    def run(data_path: str, output_path: str, parameters_file: str, checkpoint_path: str) -> None:
        # Load parameters from the paramters file
        with open(parameters_file, "r") as stream:
            parameters = yaml.safe_load(stream)

        application_name = parameters["APPLICATION_NAME"]
        application_version = parameters["APPLICATION_VERSION"]
        run_inference(data_path, output_path, checkpoint_path,
                      application_name, application_version)


# Don't delete this; if only one named command is defined, typer doesn't recognize the `infer` command any more.
@app.command("example")
def run_shit(
    parameters_file: str = typer.Option(..., "--parameters_file")
):
    print(parameters_file)


@app.command("infer")
def infer(
    data_path: str = typer.Option(..., "--data_path"),
    output_path: str = typer.Option(..., "--output_path"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
    ckpt_path: str = typer.Option(..., "--checkpoint_path")
):
    if not Path(ckpt_path).exists():
        print(ckpt_path)
        # For federated evaluation, model needs to be stored here
        print("WARNING: Checkpoint path not specified or doesn't exist. Using default path instead.")
        ckpt_path = "/mlcube_project/model_ckpts"
    if not Path(parameters_file).exists():
        # For federated evaluation, extra parameters need to be stored here
        print("WARNING: Parameter file not specified or doesn't exist. Using default path instead.")
        parameters_file = "/mlcube_project/parameters.yaml"
    InferTask.run(data_path, output_path, parameters_file, ckpt_path)


if __name__ == "__main__":
    app()

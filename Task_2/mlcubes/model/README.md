# FeTS Challenge task 2 - MLCube integration - Model

The FeTS challenge 2022 task 2 focuses on how segmentation methods can learn from multi-institutional datasets to be robust to distribution shifts at test-time, effectively solving a domain generalization problem. In this repository, you can find information on the container submission.

In the FeTS challenge task 2, participants can submit their solution in the form of an MLCube docker image. Note that we do not impose restrictions on the participants how they train their model nor how they perform inference, as long as the resulting algorithm is compatible with the interface described here. After training a model, the following steps are required to submit it:

1. Update the MLCube template with your custom code and dependencies ([guide below](#how-to-modify-this-project)).
2. Build and test the docker image as described [below](#task-execution).
3. Submit the container as described on the [challenge website](https://www.synapse.org/Synapse:syn28546456/wiki/630620).

To make sure that the containers submitted by the participants also run successfully on the remote institutions in the FeTS federation, we will offer functionality tests on a few toy cases. Details are provided in the [challenge website](https://www.synapse.org/Synapse:syn28546456/wiki/630620). Note that we will internally convert the submitted docker images into singularity images before running the evaluation.

## Project setup

Please follow these steps to get started:

- Install [docker](https://docs.docker.com/engine/install/). You may also have to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian) for GPU-support.
- [Install MLCube](https://mlcommons.github.io/mlcube/getting-started/) (with docker runner) to a virtual/conda environment of your choice. For example:

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
```

<!-- - (Optional) Install MLCube's singularity runner. -->
- Clone this repository

```bash
# Fetch the template from GitHub
git clone https://github.com/FETS-AI/Challenge.git
cd ./Task_2/mlcubes/model
```

To test your installation, you can run any of the commands in [this section](#task-execution).

## How to modify this project

You can change each file in this project to add your own implementation. In particular, participants will want to adapt the `Dockerfile`, `requirements.txt` and code in `project/src`. They should also add model checkpoints to their container. Each place where modifications are possible is described in some detail below. We also made a short guide for converting BraTS docker submissions to the format used in FeTS [here](#guide-for-converting-brats-submissions). Here is an overview of files in this project: 

```bash
├── mlcube
│   ├── mlcube.yaml          # MLCube configuration, defines the project, author, platform, docker and tasks.
│   └── workspace            # This folder is mounted at runtime. Note that it will be empty during fed. eval.
│       ├── data             # For example some data can be put here during local testing.
│       └── output           # Location where inference outputs are stored.
└── project
    ├── Dockerfile           # Docker file with instructions to create the image.
    ├── mlcube.py            # Python entrypoint used by MLCube, contains the logic for MLCube tasks.
    ├── model_ckpts          # Folder with checkpoint files loaded for inference.
    ├── parameters.yaml      # File with parameters used by inference procedure.
    ├── requirements.txt     # Python requirements needed to run the project inside Docker.
    └── src
        ├── my_logic.py      # Python file that contains the main logic of the project.
        └── utils
            └── utilities.py # Python utilities file that stores useful functions.
```

<details><summary><b>Requirements file </b></summary>
<p>

In this file (`requirements.txt`) you can add all the python dependencies needed for running your implementation. These dependencies will be installed during the creation of the docker image, which happens automatically when you run the ```mlcube run ...``` command.
</p>
</details>

<details><summary><b>Dockerfile </b></summary>
<p>

This file can be adapted to add your own docker labels, install some OS dependencies or to change the base docker image. Note however that we *strongly recommend* to use one of our proposed base images (`nvcr.io/nvidia/pytorch:20.08-py3` or tensorflow equivalent), to make sure your application can be executed in the federated evaluation. Note that the [pytorch (or tensorflow) version](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) inside this container is 1.7.0 (or 2.2.0), so for inference you may not be able to use features introduced in later versions, unfortunately.

</p>
</details>

<details><summary><b>MLCube yaml file </b></summary>
<p>

`mlcube.yaml` contains instructions about the docker image and platform that will be used, information about the project (name, description, authors), and also the tasks defined for the project. **Note** that this file is not submitted and changes will hence not have any effect in the official evaluation. We will use the provided template with the name of your docker image instead. To change the name of your docker image, you can use the `docker.image` field in the `mlcube.yaml` or use `docker tag` after building it.

In the existing implementation you will find the `infer` task, which will be executed in the federated evaluation. It takes the following parameters:

- Input parameters:
  - data_path: folder path containing input data
  - checkpoint_path: folder path containing model checkpoints
  - parameters_file: Extra parameters
- Output parameters:
  - output_path: folder path where output data will be stored

This task loads the input data, processes it and then saves the output result in the output_path. It also prints some information from the extra parameters.

</p>
</details>

<details><summary><b>MLCube python file </b></summary>
<p>

The `mlcube.py` file is the handler file and entrypoint described in the dockerfile. Here you can find all the logic related to how to process each MLCube task. For most challenge participants, the provided template should be usable without modifications.
Note that the *infer* task is the only one that will be executed in the evaluation pipeline.
If you still want to add a new task for your convenience (for example model training), you have to define it inside `mlcube.yaml` with its input and output parameters and then add the logic to handle this new task inside the `mlcube.py` file.

</p>
</details>

<details><summary><b>Main logic file </b></summary>
<p>

The `my_logic.py` file contains the main logic of the project; hence most of the custom implementations by challenge participants are required here. This logic file is called from the `mlcube.py` file.

*Please make sure* that your MLCube obeys the [conventions for input/output folders](#description-of-io-interface) after modification!

</p>
</details>

<details><summary><b>Utilities file </b></summary>
<p>

In the `utilities.py` file you can add some functions that will be useful for your main implementation. In this case, the functions from the utilities file are used inside the main logic file.

</p>
</details>

<details><summary><b>Model checkpoint(s) </b></summary>
<p>

This directory contains model checkpoints that are loaded for inference. The checkpoints used for a challenge submission have to be stored inside the MLCube to guarantee reproducibility. Therefore, please copy them to the `project/model_ckpts` directory, which will be copied to the docker image if you use the provided Dockerfile.
When testing your MLCube locally, different checkpoint directories can be passed to an existing MLCube without rebuilding the image, as described in the [example section](#tasks-execution)). 

</p>
</details>

<details><summary><b>Parameters file </b></summary>
<p>

This file (`parameters.yaml`) contains all extra parameters that aren't files or directories. For example, here you can place all the hyperparameters that you will use for training a model. The parameters used for a challenge submission have to be stored inside the MLCube to guarantee reproducibility. Therefore, please copy the final paramters to the `project/parameters.yaml` file, which will be copied to the docker image if you use the provided Dockerfile.
When testing your MLCube locally, different parameter files can be passed to an existing MLCube without rebuilding the image, as described in the [example section](#tasks-execution)).

</p>
</details>

More information on the internals of MLCube can be found in the official [git repository](https://github.com/mlcommons/mlcube) or [documentation](https://mlcommons.github.io/mlcube/). 

## Task execution

Here we describe the simple commands required to build and run individual MLCubes, which is useful for debugging your submission.
To run the complete evaluation pipeline (including toy data preparation and scoring), follow the steps [here](../../README.md#how-to-run-the-evaluation-pipeline-locally).
Note that we use docker-based MLCubes for development, which are converted automatically to singularity images before the official evaluation.

First, make sure that you are still in the `mlcube` folder. To run the `infer` task specified by the MLCube:

```bash
# Run main task
mlcube run --mlcube=mlcube.yaml --task=infer
```

By default, this will try to pull the image specified in the `docker` section of `mlcube.yaml` from dockerhub. To rebuild the docker based on local modifications, challenge participants should run:

```Bash
# Run main task and always rebuild
mlcube run --mlcube=mlcube.yaml --task=infer -Pdocker.build_strategy=always
```

You can pass parameters defined in the `mlcube.yaml` file to the MLCube like this:

```Bash
# Run main task with custom parameters
mlcube run --mlcube=mlcube.yaml --task=infer data_path=/path/to/data checkpoint_path=/path/to/checkpoints
```

where paths have to be specified as absolute paths. Refer to [this section](#mlcube-yaml-file) which parameters are supported. Note however, that only `data_path` and `output_path` will be available during federated evaluation.

If you want to build the docker image without running it, you can use

```Bash
# Only build without running a task
mlcube configure --mlcube=mlcube.yaml -Pdocker.build_strategy=always
```

## Description of IO-interface

At inference, the MLCube gets the path to the test data as input. All cases will be organized in the following structure:

```
data/ # this path is passed for inference
│
└───Patient_001 # case identifier
│   │ Patient_001_brain_t1.nii.gz
│   │ Patient_001_brain_t1ce.nii.gz
│   │ Patient_001_brain_t2.nii.gz
│   │ Patient_001_brain_flair.nii.gz
│   
└───Pat_JohnDoe # other case identifier
│   │ ...
```

Furthermore, predictions for test cases should be placed in an output directory and named as follows: `<case-identifier>.nii.gz`
An example for loading images and saving segmentations is included in [`my_logic.py`](project/src/my_logic.py).


## Guide for converting BraTS submissions

This section is supposed to help teams that already created a docker submission for BraTS 2021 with converting it so that it's a valid FeTS task-2 submission. The first step is to download [this folder](.) and copy your code to `project/src`. Then, you will need to modify a few files:

- `mlcube.py`: You can write a simple wrapper that basically calls your original inference code for each test case. This could look similar to this:
  ```python
  # ...

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
      
      for idx, subject_dir in enumerate(Path(data_path).iterdir()):
          if subject_dir.is_dir():
              subject_id = subject_dir.name
              print("Processing subject {}".format(subject_id))
              # run code from original BraTS submission. 
              # TODO Make sure your code can handle input/output paths as arguments: --input and --output. Also make sure outputs from previous runs in the output are not overwritten
              single_case_cmd = ["<insert_your_entrypoint>", "--input", str(subject_dir), "--output", str(output_path)]
              subprocess.run(single_case_cmd, check=True)
  ```
  If your original entrypoint is a python script, you can of course also import it in `mlcube.py` instead of using a subprocess. It is important to keep the interface of the `infer` command unchanged.

- `requirements.txt`: Update the python requirements.

- `Dockerfile`: Merge your Dockerfile with the one provided in [`project/Dockerfile`](./project/Dockerfile). It's important to make `mlcube.py` the entrypoint now, as in our Dockerfile. If possible, you should try to use the base image (`FROM` instruction) we suggest, to guarantee your container runs on various GPU setups.

- `model_ckpts`: Your model checkpoints have to be embedded in the docker image. Copy them here before building the image and make sure they are found by your script inside the container.

- `mlcube.yaml`: Insert your custom image name in the `docker.image` field.

After these changes, you should be able to run tests using the commands from [this section](#task-execution). Once these run without error, you're ready to [submit](https://www.synapse.org/Synapse:syn28546456/wiki/630620)!

## Project workflow

![MLCube workflow](https://i.imgur.com/qXRp3Tb.png)
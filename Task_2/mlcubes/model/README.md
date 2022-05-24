# FeTS Challenge task 2 - MLCube integration - Model

The FeTS challenge 2022 task 2 focuses on how segmentation methods can learn from multi-institutional datasets to be robust to distribution shifts at test-time, effectively solving a domain generalization problem. In this repository, you can find information on the container submission.

In the FeTS challenge task 2, participants can submit their solution in the form of an MLCube docker image. Note that we do not impose restrictions on the participants how they train their model nor how they perform inference, as long as the resulting algorithm is compatible with the interface described here. After training a model, the following steps are required to submit it:

1. Update the MLCube template with your custom code and dependencies ([guide below](#how-to-modify-this-project)).
2. Build and test the docker image as described [below](#task-execution).
3. Submit the container as described on the [challenge website](https://www.synapse.org/#!Synapse:syn28546456/wiki/617255).

To make sure that the containers submitted by the participants also run successfully on the remote institutions in the FeTS federation, we will offer functionality tests on a few toy cases. Details are provided in the [challenge website](https://www.synapse.org/#!Synapse:syn28546456/wiki/617255). Note that we will internally convert the submitted docker images into singularity images before running the evaluation.

## Project setup

Please follow these steps to get started:
<!-- TODO singularity stuff once it is ready -->
- Install [docker](https://docs.docker.com/engine/install/). You may also have to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian) for GPU-support.
<!-- - (Optional) Install [singularity](https://sylabs.io/guides/latest/user-guide/quick_start.html#quick-installation-steps). Only required if you want to test docker-to-singularity conversion yourself. -->
- [Install MLCube](https://mlcommons.github.io/mlcube/getting-started/) (with docker runner) to a virtual/conda environment of your choice. For example:

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
```

<!-- - (Optional) Install MLCube's singularity runner. -->
- Clone this repository
<!-- TODO Update this once merged -->

```bash
# Fetch the examples from GitHub
git clone https://github.com/mlcommons/mlcube_examples
cd ./mlcube_examples/fets/model/mlcube
```

To test your installation, you can run any of the commands in [this section](#task-execution).

## Important files

These are the most important files on this project:

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

## How to modify this project

You can change each file described above in order to add your own implementation. In case you need more information on the internals of MLCube, check out the official [git repository](https://github.com/mlcommons/mlcube) or [documentation](https://mlcommons.github.io/mlcube/).

<details><summary><b>Requirements file </b></summary>
<p>

In this file (`requirements.txt`) you can add all the python dependencies needed for running your implementation. These dependencies will be installed during the creation of the docker image, which happens automatically when you run the ```mlcube run ...``` command.
</p>
</details>

<details><summary><b>Dockerfile </b></summary>
<p>

This file can be adapted to add your own docker labels, install some OS dependencies or to change the base docker image. Note however that we *strongly recommend* to use one of our proposed base image, to make sure your application can be executed in the federated evaluation. Inside the file you can find some information about the existing steps.

</p>
</details>

<details><summary><b>MLCube yaml file </b></summary>
<p>

`mlcube.yaml` contains instructions about the docker image and platform that will be used, information about the project (name, description, authors), and also the tasks defined for the project. **Note** that this file is not submitted and changes will hence not have any effect in the official evaluation. We will use the provided template with the name of your docker image instead.

In the existing implementation you will find the `infer` task, which will be executed in the federated evaluation. It takes the following parameters:

- Input parameters:
  - data_path: folder path containing input data
  - checkpoint_path: folder path containing model checkpoints
  - parameters_file: Extra parameters
- Output parameters:
  - output_folder: folder path where output data will be stored

This task loads the input data, processes it and then saves the output result in the output_folder. It also prints some information from the extra parameters.

</p>
</details>

<details><summary><b>MLCube python file </b></summary>
<p>

The `mlcube.py` file is the handler file and entrypoint described in the dockerfile. Here you can find all the logic related to how to process each MLCube task. For most challenge participants, the provided template should be usable without modifications.
If you want to add a new task first you must define it inside the `mlcube.yaml` file with its input and output parameters and then you need to add the logic to handle this new task inside the `mlcube.py` file.

</p>
</details>

<details><summary><b>Main logic file </b></summary>
<p>

The `my_logic.py` file contains the main logic of the project; hence most of the custom implementations by challenge participants are required here. This logic file is called from the `mlcube.py` file.

*Please make sure* that your MLCube obeyes the [conventions for input/output folders](#description-of-io-interface) after modification!

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

## Task execution

Here we describe the simple commands required to build and run MLCubes. Note that we use docker-based MLCubes for development, which are converted automatically to singularity images before the official evaluation.

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

<!-- TODO add singularity part once it's ready -->
<!-- To use the Singularity runner instead, add the flag `--platform=singularity`:

```bash
# Run main task with singularity runner
mlcube run --mlcube=mlcube.yaml --task=infer --platform=singularity
```

Note that you need singularity installed and the MLCube singularity runner to run your MLCube with singularity. -->

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


## Project workflow

![MLCube workflow](https://i.imgur.com/qXRp3Tb.png)
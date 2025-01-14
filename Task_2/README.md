_Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Please make sure that your usage of this code is in compliance with the code license:_
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

---

# Task 2: Generalization "in the wild"

This tasks focuses on segmentation methods that can learn from multi-institutional datasets how to be robust to cross-institution distribution shifts at test-time, effectively solving a domain generalization problem. In this repository, you can find information on the container submission and ranking for task 2 of the FeTS challenge 2021. We provide:

- [MLCube (docker) template](mlcubes/model): This is a guide how to build a container submission. For more details on how to submit to task 2 of the FeTS challenge 2022, see the [challenge website](https://www.synapse.org/#!Synapse:syn28546456/wiki/617255).
- [MLCubes fore evaluation pipeline](mlcubes): These are used for running the evaluation pipeline. Participants should not modify them, as they are just provided for transparency of the official evaluation.
- Code that is used to compute the final [ranking](ranking)

The requirements of these components are described in the readme files of the respective folders. Below, you find information on how to prepare a submission and run our sanity check on them. Please also note the [hardware constraints](#hardware-constraints-for-submissions) submissions have to obey.

## How to prepare your submission container

You need to modify the MLCube template we provide. Details are described [here](mlcubes/model).

## How to run the evaluation pipeline locally

Once you have prepared your submission and pushed it to [synapse](https://www.synapse.org/#!Synapse:syn28546456/wiki/617255), it's possible to run the official evaluation pipeline on toy test cases for sanity-checking your submission. To do so, please follow these steps:

1. Make sure [docker](https://docs.docker.com/engine/install/) is installed. It is required for downloading the pipeline components. Then, log into the synapse docker registry: `docker login  docker.synapse.org` using your synapse username and password
1. [Download](https://hub.dkfz.de/s/Ctb6bQ7mbiwM6Af) the medperf environment folder and unpack it:
    ```bash
    mkdir ~/.medperf_fets
    tar -xzvf /path/to/downloaded/medperf_env.tar.gz -C ~/.medperf_fets
    ```
2. Setup python environment (install MedPerf):
    ```bash
    # Optional but recommended: use conda or virtualenv
    conda create -n fets_medperf python=3.9 pip=24.0
    conda activate fets_medperf
    # Actual installation. Important: Please use the branch below
    cd ~
    git clone https://github.com/mlcommons/medperf.git && \
        cd medperf/cli && \
        git checkout fets-challenge && \
        pip install -e .
    ```
4. Run the sanity check with docker:
    ```
    medperf --log=debug --no-cleanup --platform=docker test -b 1
    ```
    Above will run the default model defined in this [folder](mlcubes/model/mlcube/). To use your local model, please specify its path with -m:
    ```
    MODEL_PATH=/path/to/local/mlcube/folder
    medperf --log=debug --no-cleanup test -b 1 -m $MODEL_PATH
    ```
    Note that the folder passed with `-m` needs to contain an `mlcube.yaml`, which is used to pull the docker image and set runtime arguments.

The results and logs from your local test run are located in the `~/.medperf/results` and `~/.medperf/logs` folder, respectively. They can be compared to the test run executed on the organizers' infrastructure to guarantee reproducibility. Making a submission on [synapse](https://www.synapse.org/#!Synapse:syn28546456/wiki/617255) will trigger a test run through the organizers. Note that we will convert the docker images to singularity on our end. If you would like to run with singularity as well, please ask a question in the [forum](https://www.synapse.org/#!Synapse:syn28546456/discussion/default).

Note that the toy test cases are part of the FeTS 2022 training data and the same [data usage agreements](https://www.synapse.org/#!Synapse:syn28546456/wiki/617246) apply.

## Hardware Constraints for Submissions

In the testing phase of Task 2, we are going to perform a federated evaluation on multiple remote institutions with limited computation capabilities. To finish the evaluation before the MICCAI conference, we have to restrict the inference time of the submitted algorithms. As the number of participants is not known in advance, we decided for the following rules in that regard:

- We will perform a test run of the submission on three toy test cases (shipped with the MedPerf environment) on a system with one GPU (11GB) and 40 GB RAM.
- For each submission, we are going to check if the algorithms produces valid outputs on the toy test cases. Submissions that exit with error are invalid.
- Participants are allowed to do their own memory management to fit a larger algorithm, but there will be a timeout of `num_cases * 180` seconds on the inference time.
<!-- - After conversion to a singularity image file, each submission has to be smaller than 12GB. Participants will be notified if this limit is exceeded during the test run. -->

## Common Problems

Problems related to docker -> singularity conversion. There are some cases in which a docker submission can be run without errors by the submitter, but the same container causes errors on the organizers' end (because we convert them to singularity):
- `WORKDIR` not set in singularity: If `WORKDIR` is used in the Dockerfile, this can result in `FileNotFoundError` when we run your submission with singularity. To avoid this, please use only absolute paths in your code. Also the entrypoint of the container should use an absolute path to your script.
- Limited tmpfs space in singularity: Often causes errors like `OSError: [Errno 28] No space left on device`. Solution: Please make sure you write files only to the `output_path` passed to `mlcube.py`. Temporary files can be saved in a sub-directory of `output_path`, for example.
- User inside singularity containers isn't root: This can lead to `PermissionError` when reading files from the file system like model checkpoints. Make sure that all files that need to be read from inside the container can be read by *all users*, either before copying them in the Dockerfile or adding chmod commands to the Dockerfile.

Any other Errors ? Feel free to contact us: [forum](https://www.synapse.org/#!Synapse:syn28546456/discussion/default)

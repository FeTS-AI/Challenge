# Guide for singularity containers

This folder contains examples for how to prepare a container submission for task 2:
- A simplistic example using a dummy prediction algorithm (`_simple`)
- A baseline that uses nnUNet to make predictions (`_nnunet`)

Each features a definition file (`.def`), which is used to *build* the container, and a python script that performs the actual prediction (`prediction_*.py`) when the container is *run*.

Instructions on how to build and run a singularity container are given below. For a comprehensive introduction to singularity, [this tutorial](https://singularity-tutorial.github.io/) (external resource) is recommended. Note that singularity has to be installed [(instructions)](https://sylabs.io/guides/3.7/user-guide/quick_start.html#quick-installation-steps). You can build and run an example from above to check that your installation is working. If you want to build the nnUNet container yourself, you have to install [nnunet](https://github.com/MIC-DKFZ/nnUNet#installation) and download the pretrained models.

A ready-to-use nnUNet container, which was trained on the BraTS 2020 data (without taking into account the partitioning), can be downloaded from [here](https://cloud.sylabs.io/library/mzenk/fets/nnunet-brats2020).

## Building a container
A definition file is required to build a container. You can start from the example `.def`-files provided in this repo to create your own and then run
```
sudo singularity build container_simple.sif container_simple.def
```
where `container_simple` should be replaced with your corresponding file names. To avoid building with sudo, you can use singularity's `--fakeroot` option, but you may have to configure this first (`sudo singularity config fakeroot --add <username>`). Make sure that all files that are required at test-time are copied to the container when you build it. The container will not have access to your filesystem and environment when evaluated in the testing phase!

**Note:** Please use a bootstrap image based on CUDA 11.0 to make sure that the application can run on systems with Ampere architecture, too. Our recommendations are:
- `nvcr.io/nvidia/pytorch:20.08-py3` (pytorch)
- `nvcr.io/nvidia/tensorflow:20.08-tf2-py3` or `nvcr.io/nvidia/tensorflow:20.08-tf1-py3` (tensorflow)


Tip for debugging: use the `--sandbox` option, as described [here](https://singularity-tutorial.github.io/03-building/).

## Running a container
Once you have successfully built your container, it's time to test it on some data. This is the command that will be executed at test-time:
<!-- (see also [`run_submission.py`](../scripts/run_submission.py)) -->
```
DATA_DIR=/path/to/test/data/dir
OUT_DIR=/path/to/output/dir
export SINGULARITY_BINDPATH="$DATA_DIR:/data:ro,$OUT_DIR:/out_dir:rw"
singularity run -C --writable-tmpfs --net --network=none --nv container_simple.sif -i /data -o /out_dir
```
`SINGULARITY_BINDPATH` makes sure that all necessary data files can be accessed from within the container as [bind](https://singularity-tutorial.github.io/05-bind-mounts/) [mounts](https://sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html) (pattern `src:dest:read-write-mode`). Please insert the path to your data and output directory here before running the script and make sure the data is in the form described [here](../README.md#requirements)!

Description of the `singularity run` options (see also `singularity run --help`):
- `--C` : Restrict access to host filesytem and environment to a minimum
- `--writable-tmpfs` : Allows writing to a in-memory temporary filesystem*
- `--net --network=none` : No network access
- `--nv` : use GPU (nvidia) support

We recommend using above options when performing local tests to avoid problems during the testing phase.

Tip for debugging: `singularity shell` instead of `singularity run` starts an interactive shell inside the container.

*) Note that this filesystem is usually rather small, so we recommend to store any intermediate outputs in the output directory instead and clean up after inference is complete.

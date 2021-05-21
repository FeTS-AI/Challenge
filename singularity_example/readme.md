# Guide for singularity example

This repository contains examples for how to prepare a container submission for task 2:
- A simplistic example using a dummy prediction algorithm (`_simple`)
- An example using nnUNet-models to make predictions (`_nnunet`)
Each features a definition file (`.def`) and a python script that calls the actual prediction function (`prediction_*.py`).

Instructions on how to build and run a singularity container are given below. For a comprehensive introduction to singularity, [this tutorial](https://singularity-tutorial.github.io/) (external resource) is recommended. Note that singularity has to be installed [(instructions)](https://sylabs.io/guides/3.7/user-guide/quick_start.html#quick-installation-steps).

## Building a container
```
singularity build --fakeroot container_simple.sif container_simple.def
```
With `--fakeroot`, you don't need to build with sudo (security option).

Tip for debugging: use the `--sandbox` option, as described [here](https://singularity-tutorial.github.io/03-building/).

## Running a container
This is the command that will be executed at test time:
```
DATA_DIR=/path/to/test/data/dir
OUT_DIR=/path/to/output/dir
export SINGULARITY_BINDPATH="$DATA_DIR:/data:ro,$OUT_DIR:/out_dir:rw"
singularity run -C --writable-tmpfs --net --network=none --nv container_simple.sif -i /data -o /out_dir
```
The environment variable makes sure that all necessary data files can be accessed from within the container as [bind](https://singularity-tutorial.github.io/05-bind-mounts/) [mounts](https://sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html) (pattern `src:dest:read-write-mode`). Please insert the path to your data and output directory here before running the script and make sure the data is in the form described [here](../readme.md#requirements)!

Description of the `singularity run` options (see also `singularity run --help`):
- `--C` : Restrict access to host filesytem and environment to a minimum
- `--writable-tmpfs` : Allows writing to a in-memory temporary filesystem*
- `--net --network=none` : No network access
- `--nv` : use experimental GPU (nvidia) support

*) Note that this filesystem is usually rather small, so we recommend to store any intermediate outputs in the output directory instead.

Tip for debugging: `singularity shell` instead of `singularity run` starts an interactive shell inside the container.

# Guide for singularity example

This repository contains examples for how to prepare a container submission for task 2:
- A simplistic example using a dummy prediction algorithm (`_simple`)
- An example using nnUNet-models to make predictions (`_nnunet`)
Each features a definition file (`.def`) and a python script that calls the actual prediction function (`prediction_*.py`).

Instructions how to build and run a singularity container are given below. For a comprehensive introduction to singularity, [this tutorial](https://singularity-tutorial.github.io/) (external resource) is recommended.

## Prerequisites
Singularity has to be installed [(instructions)](https://sylabs.io/guides/3.7/user-guide/quick_start.html#quick-installation-steps)

## Building a container
```
singularity build --fakeroot container_simple.sif container_simple.def
```
With `--fakeroot`, you don't need to build with sudo (security option).

Tip for debugging building: use the `--sandbox` option

## Running a container
This is the command that will be executed at test time:
```
export SINGULARITY_BINDPATH="/path/to/test/data:/data:ro,/path/to/output/dir:/out_dir:rw"
singularity run -C --writable-tmpfs --net --network=none --nv container_simple.sif -i /data -o /out_dir
```
The environment variable makes sure that all necessary data files can be accessed from within the container as bind mounts (pattern `src:dest:read-write-mode`).
Description of the `singularity run` options (see also `singularity run --help`):
- `--C` : Restrict access to host filesytem and environment to a minimum
- `--writable-tmpfs` : Allows writing to a in-memory temporary filesystem
- `--net --network=none` : No network access
- `--nv` : use experimental GPU (nvidia) support

Tip for debugging runscript (defined in the .def file): `singularity shell` instead of `singularity run` gives an interactive shell.


# To do's
- The tmpfs when using writable-tmpfs seems to be very small. Not sure how to fix it, here are some ideas: 
  https://github.com/hpcng/singularity/issues/5718
  https://groups.google.com/a/lbl.gov/g/singularity/c/eq-tLo2SewM

  *update 21/04/21*: I'm not sure any more where this causes a problem. My container does not need the tmpfs.
  *update 18/05/21*: Maybe add a warning on this, discuss with David
- bootstrap image: Could there be problems with the nvidia driver? 
  From: nvcr.io/nvidia/pytorch:21.04-py3 worked on my workstation but I'm not sure if this was just because of "enhanced compatibility in the CUDA 11 versions.
  -> May have to specify minimum/maximum driver version as "HW requirement" for collaborators/participants
- team names: Teams  should add their team-identifier (letters only, no symbols) as a tag maybe? Or, depending on the submission platform, I will also assign them names (?)
- weird message upon container execution? (see https://forums.developer.nvidia.com/t/change-mofed-version-usage-example-message-with-tensorflow-ngc-container/124178)

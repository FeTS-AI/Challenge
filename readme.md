# Task 2: Generalization "in the wild"

This repository includes information on the container submission and ranking for task 2 of the FeTS challenge 2021. It is structured as follows

- `ranking`: Code for performing the final ranking
- `scripts`: Code for running containers, both in the participant's environment and in the final federated testing environment
- `singularity_example`: Guide how to build the container submission with examples

Details about the submission process are given on the challenge [website](https://fets-ai.github.io/Challenge/).

## Requirements
Singularity has to be installed to create a container submission [(instructions)](https://sylabs.io/guides/3.7/user-guide/quick_start.html#quick-installation-steps).

Python 3.6 is required to run the scripts.

## Test your singularity container

If you don't know how to create a container, have a look at the guide [here](singularity_example/readme.md).
Once you built your container, you can run the testing script as follows:

```bash
python scripts/test_container.py container.sif -i /path/to/data [-o /path/to/output_dir]
```

This will run the container on the data in the input folder (`-i`) and (optionally) save the outputs in the output folder (`-o`). It will also do a sanity check on your outputs, so that you are warned if something is not as it should be.

**Note** The current version of this script is preliminary. It will be updated soon.
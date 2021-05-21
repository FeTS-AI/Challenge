# Task 2: Generalization "in the wild"

This repository includes information on the container submission and ranking for task 2 of the FeTS challenge 2021. It is structured as follows

- `singularity_example`: Guide how to build the container submission with examples
- `scripts`: Code for running containers, both in the participant's environment and in the federated testing environment
- `ranking`: Code for performing the final ranking

Details about the submission process are given on the [challenge website](https://fets-ai.github.io/Challenge/).

## Requirements
Singularity has to be installed to create a container submission [(instructions)](https://sylabs.io/guides/3.7/user-guide/quick_start.html#quick-installation-steps).

Python 3.6 or higher is required to run the scripts.

The examples in this repo assume the following data folder structure, which will also be present at test-time:
```
data/ # this should be passed for inference
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

## How to build a singularity container

If you don't know how to create a container, have a look at the guide in the [singularity_example](singularity_example/readme.md).

## Test your own container

Once you have built your container, you can run the testing script as follows:

```bash
python scripts/test_container.py container.sif -i /path/to/data [-o /path/to/output_dir]
```

This will run the container on the data in the input folder (`-i`) and (optionally) save the outputs in the output folder (`-o`). It will also do a sanity check on your outputs, so that you are warned if something is not as it should be.

**Note** The current version of this script is preliminary. It will be updated soon.
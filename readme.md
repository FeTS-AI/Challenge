# Task 2: Generalization "in the wild"

This repository includes information on the container submission and ranking for task 2 of the FeTS challenge 2021. It is structured as follows

- `singularity_example`: Guide how to build the container submission with examples
- `scripts`: Code for running containers, both in the participant's environment and in the federated testing environment
- `ranking`: Code for performing the final ranking

In the FeTS challenge task 2, participants can submit their solution in the form of a [singularity container](https://sylabs.io/guides/3.7/user-guide/index.html). After training a model, the following steps are required to submit it:

1. Write a container definition file and an inference script.
2. Build a singularity container for inference using above files and the final model weights.
3. Upload the container to the submission platform (tbd)

Details for steps 1 and 2 are given in the guide in the [singularity_example](singularity_example/readme.md). Instructions for step 3 will follow soon.

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
Furthermore, predictions for test cases should be placed in one output directory and named like this: `CASE-ID_TEAM-NAME_seg.nii.gz`

## Test your own container

Once you have built your container, you can run the testing script as follows:

```bash
python scripts/test_container.py container.sif -i /path/to/data [-o /path/to/output_dir]
```

This will run the container on the data in the input folder (`-i`) and (optionally) save the outputs in the output folder (`-o`). It will also do a sanity check on your outputs, so that you are warned if something is not as it should be.

**Note** The current version of this script is preliminary. It will be updated soon.
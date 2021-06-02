_Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Please make sure that your usage of this code is in compliance with the code license:_
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

---

# Task 2: Generalization "in the wild"

This tasks focuses on how segmentation methods can learn from multi-institutional datasets how to be robust to distribution shifts at test-time, effectively solving a domain generalization problem. In this repository, you can find information on the container submission and ranking for task 2 of the FeTS challenge 2021. It is structured as follows:

- [`singularity_example`](singularity_example): Guide how to build the container submission with examples
- [`scripts`](scripts): Scripts for running containers, both in the participant's environment and in the federated testing environment
- [`ranking`](ranking): Code for performing the final ranking

In the FeTS challenge task 2, participants can submit their solution in the form of a [singularity container](https://sylabs.io/guides/3.7/user-guide/index.html). Note that we do not impose restrictions on the participants how they train their model nor how they perform inference, as long as the resulting algorithm can be built into a singularity container with the simple interface described in `singularity_example`. Hence, after training a model, the following steps are required to submit it:

1. Write a container definition file and an inference script.
2. Build a singularity container for inference using above files and the final model weights.
3. Upload the container to the submission platform.

Details for steps 1 and 2 are given in the guide in the [singularity_example](singularity_example). Instructions for step 3 will follow soon.

## Requirements
Singularity has to be installed to create a container submission [(instructions)](https://sylabs.io/guides/3.7/user-guide/quick_start.html#quick-installation-steps).

Python 3.6 or higher is required to run the scripts in `scripts`.

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
Furthermore, predictions for test cases should be placed in an output directory and named like this: `<case-identifier>_seg.nii.gz`

<!-- ## Test your own container

Once you have built your container, you can run the testing script as follows:

```bash
python scripts/test_container.py container.sif -i /path/to/data [-o /path/to/output_dir]
```

This will run the container on the data in the input folder (`-i`) and (optionally) save the outputs in the output folder (`-o`). It will also do a sanity check on your outputs, so that you are warned if something is not as it should be.

To compute the segmentation metrics as it is done during the testing phase, the [CaPTk CLI](https://cbica.github.io/CaPTk/BraTS_Metrics.html) can be used. Please refer to their website for installation and usage instructions.

**Note** The current version of this script is preliminary. It will be updated soon. -->
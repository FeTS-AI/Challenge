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

Details for steps 1 and 2 are given in the guide in the [singularity_example](singularity_example). Regarding step 3, each participating team will be provided a gitlab project where they can upload their submission. A few simple steps are necessary for that:

1. Register for the challenge as described on the [challenge website](https://fets-ai.github.io/Challenge/) (if not already done).
2. Sign up at [https://gitlab.hzdr.de/](https://gitlab.hzdr.de/) **using the same email address as in step 1** by either clicking *Helmholtz AAI* (login via your institutional email) or via your github login. Both buttons are in the lower box on the right.
3. Send an email to [challenge@fets.ai](mailto:challenge@fets.ai), asking for a Task 2-gitlab project and stating your gitlab handle (@your-handle) and team name. We will create a project for you and invite you to it within a day.
4. Follow the instructions in the newly created project to make a submission.

To make sure that the containers submitted by the participants also run successfully on the remote institutions in the FeTS federation, we offer functionality tests on a few toy cases. Details are provided in the gitlab project.

## Requirements
Singularity has to be installed to create a container submission [(instructions)](https://sylabs.io/guides/3.7/user-guide/quick_start.html#quick-installation-steps).

Python 3.6 or higher is required to run the scripts in `scripts`. Make sure to install the requirements (e.g. `pip install -r requirements.txt`), preferably in a virtual/conda environment.

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

## Test your container

Once you have built your container, you can run the testing script as follows:

```bash
python scripts/test_container.py container.sif -i /path/to/data [-o /path/to/output_dir -l /path/to/label_dir]
```

This will run the container on the data in the input folder (`-i`), which should be formatted as described in the [requirements](#requirements), and save the outputs in the output folder (`-o`); without the latter option, outputs will be deleted at the end. This script will also report the execution time and do a sanity check on your outputs, so that you are warned if something is not as it should be. To test the functionality and runtime of your container on a standardized setup, please make a submission via gitlab, as described in the first section. From gitlab you can also get a small reference dataset with the correct folder structure and naming.

If labels are provided, this script also computes metrics for each test case and saves them in the output folder. **Note**, however, that these metrics are just for sanity checks and will be computed differently during the testing phase. Specifically, the [CaPTk library](https://cbica.github.io/CaPTk/BraTS_Metrics.html) will be used in the test phase evaluation. If you would like to try it on your predictions, please refer to their website for installation and usage instructions.

## Submission Rules
In the testing phase of Task 2, we are going to perform a federated evaluation on multiple remote institutions with limited computation capabilities. To finish the evaluation before the MICCAI conference, we have to restrict the inference time of the submitted algorithms. As the number of participants is not known in advance, we decided for the following rules in that regard:
- For each final submission, we are going to check the validity of the algorithm output and measure the execution time of the container on a small dataset using a pre-defined Hardware setup (CPU: E5-2620 v4, GPU: RTX 2080 Ti 10.7GB, RAM: 40GB).
- Each submission is given **180 seconds per case** to produce a prediction (we will check only the total runtime for all cases, though). Submissions that fail to predict all cases within this time budget will not be included in the federated evaluation.
- If the number of participants is extremely high, we reserve the right to limit the number of participants in the final MICCAI ranking in the following way: Algorithms will be evaluated on the federated test set in the chronological order they were submitted in. This means the later an algorithm is submitted, the higher is the risk it cannot be evaluated on all federated test sets before the end of the testing phase. Note that this is a worst-case rule and we will work hard to include every single valid submission in the ranking.

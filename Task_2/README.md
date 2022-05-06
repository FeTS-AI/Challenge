_Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Please make sure that your usage of this code is in compliance with the code license:_
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

---

# Task 2: Generalization "in the wild"

This tasks focuses on segmentation methods that can learn from multi-institutional datasets how to be robust to cross-institution distribution shifts at test-time, effectively solving a domain generalization problem. In this repository, you can find information on the container submission and ranking for task 2 of the FeTS challenge 2021. We provide:

- [MLCube (docker) template](https://github.com/mlcommons/mlcube_examples/tree/master/fets/model): This is a guide how to build a container submission. For more details on how to submit to task 2 of the FeTS challenge 2022, see the [challenge website](https://www.synapse.org/#!Synapse:syn28546456/wiki/617255).
- A [script](scripts/generate_toy_test_cases.py) to extract "toy test cases" from the official training data. These can be used for verifying segmentation performance in functionality tests prior to the final submission. More details on the [challenge website](https://www.synapse.org/#!Synapse:syn28546456/wiki/617255).
- Code that is used to compute the final [ranking](ranking)

## Requirements

In order to run the `generate_toy_test_cases.py` script, you need the official [challenge training data](https://www.synapse.org/#!Synapse:syn28546456/wiki/617246). Also, Python 3.6 or higher is required.

The ranking code requirements are described [here](ranking).

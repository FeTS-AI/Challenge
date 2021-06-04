# FeTS 2021 Challenge Task 1
Task 1 (**"Federated Training"**) aims at effective weight aggregation methods for the creation of a consensus model given a pre-defined segmentation algorithm for training, while also (optionally) accounting for network outages.

Please ask any additional questions in our discussion pages on our github site and we will try to update this README.md as we identify confusions/gaps in our explanations and instructions.

## Getting started

### System requirements

1. [Git](https://git-scm.com/downloads).
2. Python with virtual environment management system: we recommend using [Anaconda](https://www.anaconda.com/products/individual).
3. **Windows- Only**: Pickle5 requires Microsoft C++ 14.0 or greater from the [C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### Instructions

1. Register for the FeTS 2021 Challenge [here](https://www.med.upenn.edu/cbica/fets/miccai2021/) and submit a data request.
2. ```git clone https://github.com/FETS-AI/Challenge.git```
3. ```cd Challenge/Task_1```
4. Create virtual environment (python 3.6-3.8): using Anaconda, a new environment can be created and activated using the following commands: 
```bash
## create venv in specific path
conda create -p ./venv python=3.7 -y
conda activate ./venv
```
5. ```pip install --upgrade pip```
6. ```pip install .```
7. ```jupyter notebook```
8. All lower-level details are in the [FeTS Challenge notebook](./FeTS_Challenge.ipynb)

## Time to Convergence Metric (formerly "communication cost")
Along with the typical DICE and Hausdorff metrics, we include a "time to convergence metric" intended to encourage solutions that converge to good scores quickly in terms of time. We simulate the time taken to run each round so that competitors don't need to be concerned with runtime optimizations such as compiled vs. interpreted code, and so that final scoring will not depend on the hardware used. This simulated time is computed in the experiment.py file and provided in the metrics output of the experiment execution.

The time to convergence metric will be computed as the area under the validation learning curve over 1 week of simulated time where the horizontal axis measures simulated runtime and the vertical axis measures the current best score, computed as the average of enhancing tumor, tumor core, and whole tumor DICE scores over the validation split of the training data.

You can find the code for the "time to convergence metric" in the experiment.py file by searching for ## CONVERGENCE METRIC COMPUTATION.

### How Simulated Time is computed
The simulated time is stochastic, and computed per collaborator, per round, with the round time equaling the greatest round time of all collaborators in the round.
 
A given collaborator's round time is computed as the sum of:
- The simulated time taken to download the shared model
- The simulated time taken to validate the shared model
- The simulated time taken to train the model (if training)
- The simulated time taken to validate that collaborator's trained model (if training)
- The simulated time taken to upload that collaborator's model update (if training)
 
During the experiment, to generate these simulated times, we first assign each collaborator four normal distrubitions representing:
1. download speed
2. upload speed
3. training speed
4. validation speed

We then draw from the appropriate distribution when generating one of the times listed above (at each round).

We assign these network and compute distributions by drawing uniform-randomly from lists of normal distributions created using timing information collected from a subset of the 50+ participants in the May FeTS initiative training of this same model. In this way, the statistics used to simulate timing information come from timing information collected over an actual federation of hospitals that trained this exact model. In particular, for each actual hospital in our subset, we collected:
1. The mean and stdev seconds to download the model
2. The mean and stdev seconds to train a batch
3. The mean and stdev seconds to validate a batch
4. The mean and stdev seconds to upload the model.

For a given collaborator, these normal distributions are constant throughout the experiment. Again, each possible timing distribution is based on actual timing information from a subset of the hospitals in the FeTS intitiative. You can find these distributions in the experiment.py file (search for ## COLLABORATOR TIMING DISTRIBUTIONS), as well as the random seed used to ensure reproducibility.

## Data Partitioning and Sharding
The FeTS 2021 data release consists of a training set and two CSV files - each providing information for how to partition the training data into non-IID institutional subsets. The release will contain subfolders for single patient records whose names have the format `FeTS21_Training_###`, and two CSV files: 
- **partitioning_1.csv**
- **partitioning_2.csv**

Each of the partitioning CSV files has two columns, `Partition_ID` and `Subject_ID`. The Subject_ID column exhausts of the patient records contained in the release. The InstitutionName column provides an integer identifier indicating to which institution the record should be assigned. The path to a partition CSV can be provided as the value of the parameter ```institution_split_csv_filename``` to the jupyter notebook function run_challenge_experiment to specify the institutional split used when running experimental federated training on your custom federation logic. A description of each of these split CSVs is provided in Table 1. We encourage participants to create and explore training performance for other non-IID splits of the training data to help in developing generalizable customizations to the federated logic that will perform well during the validation and testing phase. A third CSV is hidden from participants and defines a test partitioning to be used in the challenge testing phase. This hidden partitioning (also described in Table 1) is another refinement of the institution split, having similar difficulty level to the institution tumor size split in our own experiments using the default customization functions.

Table 1: Information for partitionings provided in the FeTS 2021 data release as well as the hidden partitioning not provided in the release (to be used in the competition testing phase).

|     Split name                      |     CSV filename                         |     Description                                                                                                                                                                                       |     Number of institutions      |
|-------------------------------------|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
|     Institution Split               |     partitioning_1.csv                |     Split of FeTS 2021 training data by originating institution.                                                                                                                                    |     17                          |
|     Institution Tumor Size Split    |     partitioning_2.csv      |     Refinement of the institution split by tumor size, further   splitting the larger institutions according to whether a record’s tumor size   fell above or below the mean size for that institution.    |     22                          |
|     Test Split                      |          - not provided -       |     Undisclosed refinement of the institution split.                                                                                                                                                  |     Hidden from participants    |




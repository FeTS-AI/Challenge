# FeTS 2022 Challenge Task 1
Task 1 (**"Federated Training"**) aims at effective weight aggregation methods for the creation of a consensus model given a pre-defined segmentation algorithm for training, while also (optionally) accounting for network outages.

Please ask any additional questions in our discussion pages on our github site and we will try to update this README.md as we identify confusions/gaps in our explanations and instructions.

## Getting started

### System requirements

1. [Git](https://git-scm.com/downloads)
2. [Git LFS](https://github.com/git-lfs/git-lfs#downloading)
2. Python with virtual environment management system: we recommend using [Anaconda](https://www.anaconda.com/products/individual).
3. **Windows- Only**: Pickle5 requires Microsoft C++ 14.0 or greater from the [C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
> * _Note: if you run into ```UnicodeDecodeError``` during installation, trying pinning ```openfl @ git+https://github.com/intel/openfl.git@v1.4``` in [setup.py](https://github.com/FeTS-AI/Challenge/blob/main/Task_1/setup.py#L31)_
4. Use CUDA 11 for your installation as CUDA 12 is not compatible with this codebase.

### Instructions --- IMPORTANT

1. Register for the FeTS 2022 Challenge [here](https://www.synapse.org/#!Synapse:syn28546456/wiki/617093) and submit a data request.
2. ```git clone https://github.com/FETS-AI/Challenge.git```
3. ```cd Challenge/Task_1```
4. ```git lfs pull```
5. Create virtual environment (python 3.6-3.8): using Anaconda, a new environment can be created and activated using the following commands: 
    ```sh
    ## create venv in specific path
    conda create -p ./venv python=3.7 -y
    conda activate ./venv
    ```
6. ```pip install --upgrade pip```
7. Install Pytorch LTS (1.8.2) for your system using [these instructions](https://pytorch.org/get-started/locally/)
8. Set the environment variable `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` (to avoid sklearn deprecation error)
9. ```pip install .``` 
> * _Note: if you run into ```ERROR: Failed building wheel for SimpleITK```, try running ```pip install SimpleITK --only-binary :all:``` then rerunning ```pip install .```_
10. ```python FeTS_Challenge.py```
11. All lower-level details are in the [FeTS Challenge python file](./FeTS_Challenge.py)
12. To view intermediate results with TensorBoard during training, you can run the following command: ```tensorboard --logdir ~/.local/workspace/logs/tensorboard```

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

### Use in Ranking

For ranking of multidimensional outcomes (or metrics), for each team, we will compute the summation of their ranks across the average of the **7** metrics (i.e., time to convergence, and Dice & Hausdorff from 3 regions of interest) described as a univariate overall summary measure. This measure will decide the overall ranking for each specific team. Notably, since all teams are ranked per patient, whereas the communication cost is only accounted once for the complete training phase, the communication cost **will be weighted** according to the number of testing subjects in order to give it **equal importance** to the quality of the tumor segmentations.


## Data Partitioning and Sharding
The FeTS 2022 data release consists of a training set and two CSV files - each providing information for how to partition the training data into non-IID institutional subsets. The release will contain subfolders for single patient records whose names have the format `FeTS2022_###`, and two CSV files: 
- **partitioning_1.csv**
- **partitioning_2.csv**

Each of the partitioning CSV files has two columns, `Partition_ID` and `Subject_ID`. The Subject_ID column exhausts of the patient records contained in the release. The InstitutionName column provides an integer identifier indicating to which institution the record should be assigned. The path to a partition CSV can be provided as the value of the parameter ```institution_split_csv_filename``` to the jupyter notebook function run_challenge_experiment to specify the institutional split used when running experimental federated training on your custom federation logic. A description of each of these split CSVs is provided in Table 1. We encourage participants to create and explore training performance for other non-IID splits of the training data to help in developing generalizable customizations to the federated logic that will perform well during the validation and testing phase. A third CSV is hidden from participants and defines a test partitioning to be used in the challenge testing phase. This hidden partitioning (also described in Table 1) is another refinement of the institution split, having similar difficulty level to the institution tumor size split in our own experiments using the default customization functions.

Table 1: Information for partitionings provided in the FeTS 2022 data release as well as the hidden partitioning not provided in the release (to be used in the competition testing phase).

|     Split name                      |     CSV filename                         |     Description                                                                                                                                                                                       |     Number of institutions      |
|-------------------------------------|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
|     Institution Split               |     partitioning_1.csv                |     Split of FeTS 2022 training data by originating institution.                                                                                                                                    |     23                          |
|     Institution Tumor Size Split    |     partitioning_2.csv      |     Refinement of the institution split by tumor size, further   splitting the larger institutions according to whether a record’s tumor size   fell above or below the mean size for that institution.    |     33                          |
|     Test Split                      |          - not provided -       |     Undisclosed refinement of the institution split.                                                                                                                                                  |     Hidden from participants    |




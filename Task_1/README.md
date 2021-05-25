# FeTS 2021 Challenge Task 1
Task 1 (**"Federated Training"**) aims at effective weight aggregation methods for the creation of a consensus model given a pre-defined segmentation algorithm for training, while also (optionally) accounting for network outages.

## Getting started

### System requirements

1. [Git](https://git-scm.com/downloads)
2. Python with virtual environment management system: we recommend using [Anaconda](https://www.anaconda.com/products/individual).
3. **Windows- Only**: Pickle5 requires Microsoft C++ 14.0 or greater from the [C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

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




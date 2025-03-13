#!/usr/bin/env python
# coding: utf-8

# # FeTS Challenge
# 
# Contributing Authors (alphabetical order):
# - Brandon Edwards (Intel)
# - Patrick Foley (Intel)
# - Alexey Gruzdev (Intel)
# - Sarthak Pati (University of Pennsylvania)
# - Micah Sheller (Intel)
# - Ilya Trushkin (Intel)


import os
import numpy as np
from fets_challenge import model_outputs_to_disc
from pathlib import Path
import shutil
import glob
from fets_challenge import run_challenge_experiment


# # Adding custom functionality to the experiment
# Within this notebook there are **four** functional areas that you can adjust to improve upon the challenge reference code:
# 
# - [Custom aggregation logic](#Custom-Aggregation-Functions)
# - [Selection of training hyperparameters by round](#Custom-hyperparameters-for-training)
# - [Collaborator training selection by round](#Custom-Collaborator-Training-Selection)
# 

# ## Experiment logger for your functions
# The following import allows you to use the same logger used by the experiment framework. This lets you include logging in your functions.

from fets_challenge.experiment import logger


# # Getting access to historical weights, metrics, and more
# The **db_iterator** parameter gives full access to all of the tensors and metrics stored by the aggregator. Participants can access these records to create advanced aggregation methods, hyperparameters for training, and novel selection logic for which collaborators should participant in a given training round. See below for details about how data is stored internally and a comprehensive set of examples. 
# 
# ## Basic Form
# Each record yielded by the `db_iterator` contains the following fields:
# 
# |                      TensorKey                     |   Tensor  |
# |:--------------------------------------------------:|:---------:|
# | 'tensor_name', 'origin', 'round', 'report', 'tags' | 'nparray' |
# 
# All records are internally stored as a numpy array: model weights, metrics, as well as hyperparameters. 
# 
# Detailed field explanation:
# - **'tensor_name'** (str): The `'tensor_name'` field corresponds to the model layer name (i.e. `'conv2d'`), or the name of the metric that has been reported by a collaborator (i.e. `'accuracy'`). The built-in validation functions used for evaluation of the challenge will be given a prefix of `'challenge_metric_\*'`. The names that you provide in conjunction with a custom validation metrics to the ```run_challenge_experiment``` function will remain unchanged.  
# - **'origin'** (str): The origin denotes where the numpy array came from. Possible values are any of the collaborator names (i.e. `'col1'`), or the aggregator.
# - **'round'** (int): The round that produced the tensor. If your experiment has `N` rounds, possible values are `0->N-1`
# - **'report'** (boolean): This field is one of the ways that a metric can be denoted; For the purpose of aggregation, this field can be ignored.
# - **'tags'** (tuple(str)): The tags include unstructured information that can be used to create complex data flows. For example, model layer weights will have the same `'tensor_name'` and `'round'` before and after training, so a tag of `'trained'` is used to denote that the numpy array corresponds to the layer of a locally trained model. This field is also used to capture metric information. For example, `aggregated_model_validation` assigns tags of `'metric'` and `'validate_agg'` to reflect that the metric reported corresponds to the validation score of the latest aggregated model, whereas the tags of `'metric'` and `'validate_local'` are used for metrics produced through validation after training on a collaborator's local data.   
# - **'nparray'** (numpy array) : This contains the value of the tensor. May contain the model weights, metrics, or hyperparameters as a numpy array.
# 

# ### Note about OpenFL "tensors"
# In order to be ML framework agnostic, OpenFL represents tensors as numpy arrays. Throughout this code, tensor data is represented as numpy arrays (as opposed to torch tensors, for example).

# # Custom Collaborator Training Selection
# By default, all collaborators will be selected for training each round, but you can easily add your own logic to select a different set of collaborators based on custom criteria. An example is provided below for selecting a single collaborator on odd rounds that had the fastest training time (`one_collaborator_on_odd_rounds`).


# a very simple function. Everyone trains every round.
def all_collaborators_train(collaborators,
                            db_iterator,
                            fl_round,
                            collaborators_chosen_each_round,
                            collaborator_times_per_round):
    """Chooses which collaborators will train for a given round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    """
    return collaborators

# this is not a good algorithm, but we include it to demonstrate the following:
    # simple use of the logger and of fl_round
    # you can search through the "collaborator_times_per_round" dictionary to see how long collaborators have been taking
    # you can have a subset of collaborators train in a given round
def one_collaborator_on_odd_rounds(collaborators,
                                   db_iterator,
                                   fl_round,
                                   collaborators_chosen_each_round,
                                   collaborator_times_per_round):
    """Chooses which collaborators will train for a given round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    """
    logger.info("one_collaborator_on_odd_rounds called!")
    # on odd rounds, choose the fastest from the previous round
    if fl_round % 2 == 1:
        training_collaborators = None
        fastest_time = np.inf
        
        # the previous round information will be index [fl_round - 1]
        # this information is itself a dictionary of {col: time}
        for col, t in collaborator_times_per_round[fl_round - 1].items():
            if t < fastest_time:
                fastest_time = t
                training_collaborators = [col]
    else:
        training_collaborators = collaborators
    return training_collaborators


# # Custom hyperparameters for training

# You can customize the hyper-parameters for the training collaborators at each round. At the start of the round, the experiment loop will invoke your function and set the hyper-parameters for that round based on what you return.
# 
# The training hyper-parameters for a round are:
# - **`learning_rate`**: the learning rate value set for the Adam optimizer
# - **`epochs_per_round`**: the number of epochs each training collaborator will train. Must be a float or None. Partial epochs are allowed, such as 0.5 epochs.
# 
# Your function will receive the typical aggregator state/history information that it can use to make its determination. The function must return a tuple of (`learning_rate`, `epochs_per_round`). For example, if you return:
# 
# `(1e-4, 2)`
# 
# then all collaborators selected based on the [collaborator training selection criteria](#Custom-Collaborator-Training-Selection) will train for `2` epochs with a learning rate of `1e-4`.
# 
# Different hyperparameters can be specified for collaborators for different rounds but they remain the same for all the collaborators that are chosen for that particular round. In simpler words, collaborators can not have different hyperparameters for the same round.

# This simple example uses constant hyper-parameters through the experiment
def constant_hyper_parameters(collaborators,
                              db_iterator,
                              fl_round,
                              collaborators_chosen_each_round,
                              collaborator_times_per_round):
    """Set the training hyper-parameters for the round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    Returns:
        tuple of (learning_rate, epochs_per_round).
    """
    # these are the hyperparameters used in the May 2021 recent training of the actual FeTS Initiative
    # they were tuned using a set of data that UPenn had access to, not on the federation itself
    # they worked pretty well for us, but we think you can do better :)
    epochs_per_round = 1
    learning_rate = 5e-5
    return (learning_rate, epochs_per_round)


# this example trains less at each round
def train_less_each_round(collaborators,
                          db_iterator,
                          fl_round,
                          collaborators_chosen_each_round,
                          collaborator_times_per_round):
    """Set the training hyper-parameters for the round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    Returns:
        tuple of (learning_rate, epochs_per_round) 
    """

    # we'll have a constant learning_rate
    learning_rate = 5e-5
    
    # our epochs per round will start at 1.0 and decay by 0.9 for the first 10 rounds
    epochs_per_round = 1.0
    decay = min(fl_round, 10)
    decay = 0.9 ** decay
    epochs_per_round *= decay    
    epochs_per_round = int(epochs_per_round)
    
    return (learning_rate, epochs_per_round)


# # Custom Aggregation Functions
# Standard aggregation methods allow for simple layer-wise combination (via weighted_mean, mean, median, etc.); however, more complex aggregation methods can be supported by evaluating collaborator metrics, weights from prior rounds, etc. OpenFL enables custom aggregation functions via the [**AggregationFunctionInterface**](https://github.com/intel/openfl/blob/fets/openfl/component/aggregation_functions/interface.py). For the challenge, we wrap this interface so we can pass additional simulation state, such as simulated time.
# 
# [**LocalTensors**](https://github.com/intel/openfl/blob/fets/openfl/utilities/types.py#L13) are named tuples of the form `('collaborator_name', 'tensor', 'collaborator_weight')`. Your custom aggregation function will be passed a list of LocalTensors, which will contain an entry for each collaborator who participated in the prior training round. The [**`tensor_db`**](#Getting-access-to-historical-weights,-metrics,-and-more) gives direct access to the aggregator's tensor_db dataframe and includes all tensors / metrics reported by collaborators. Using the passed tensor_db reference, participants may even store custom information by using in-place write operations. A few examples are included below.
# 
# We also provide a number of convenience functions to be used in conjunction with the TensorDB for those who are less familiar with pandas. These are added directly to the dataframe object that gets passed to the aggregation function to make it easier to *store* , *retrieve*, and *search* through the database so that participants can focus on algorithms instead of infrastructure / framework details.
#
# tensor_db.store:
#
#        Convenience method to store a new tensor in the dataframe.
#        Args:
#            tensor_name [ optional ] : The name of the tensor (or metric) to be saved
#            origin      [ optional ] : Origin of the tensor
#            fl_round    [ optional ] : Round the tensor is associated with
#            metric:     [ optional ] : Is the tensor a metric?
#            tags:       [ optional ] : Tuple of unstructured tags associated with the tensor
#            np.array    [ required ] : Value to store associated with the other included information (i.e. TensorKey info)
#            overwrite:  [ optional ] : If the tensor is already present in the dataframe
#                                       should it be overwritten?
#        Returns:
#            None
#
#
# tensor_db.retrieve
# 
#        Convenience method to retrieve tensor from the dataframe.
#        Args:
#            tensor_name [ optional ] : The name of the tensor (or metric) to retrieve
#            origin      [ optional ] : Origin of the tensor
#            fl_round    [ optional ] : Round the tensor is associated with
#            metric:     [ optional ] : Is the tensor a metric?
#            tags:       [ optional ] : Tuple of unstructured tags associated with the tensor
#                                       should it be overwritten?
#        Returns:
#            Optional[ np.array ]     : If there is a match, return the first row
#
# tensor_db.search
#
#        Search the tensor_db dataframe based on:
#            - tensor_name
#            - origin
#            - fl_round
#            - metric
#            -tags
#        Returns a new dataframe that matched the query
#        Args:
#            tensor_name: The name of the tensor (or metric) to be searched
#            origin:      Origin of the tensor
#            fl_round:    Round the tensor is associated with
#            metric:      Is the tensor a metric?
#            tags:        Tuple of unstructured tags associated with the tensor
#        Returns:
#            pd.DataFrame : New dataframe that matches the search query from 
#                           the tensor_db dataframe
#
# ## Converting the tensor_db to a db_iterator (to reuse aggregation methods from last year's competition)
# ### Using prior layer weights
# Here is an example of how to extract layer weights from prior round. The tag is `'aggregated'` indicates this : 
#     
#     for _, record in tensor_db.iterrows():
#             if (
#                 record['round'] == (fl_round - 1)
#                 and record['tensor_name'] == tensor_name
#                 and 'aggregated' in record['tags']
#                 and 'delta' not in record['tags']
#                ):
#                 previous_tensor_value = record['nparray']
#                 break
# 
# ### Using validation metrics for filtering
# 
#     threshold = fl_round * 0.3 + 0.5
#     metric_name = 'acc'
#     tags = ('metric','validate_agg')
#     selected_tensors = []
#     selected_weights = []
#     for _, record in tensor_db.iterrows():
#         for local_tensor in local_tensors:
#             tags = set(tags + [local_tensor.col_name])
#             if (
#                 tags <= set(record['tags']) 
#                 and record['round'] == fl_round
#                 and record['tensor_name'] == metric_name
#                 and record['nparray'] >= threshold
#             ):
#                 selected_tensors.append(local_tensor.tensor)
#                 selected_weights.append(local_tensor.weight)
# 
# ### A Note about true OpenFL deployments
# The OpenFL custom aggregation interface does not currently provide timing information, so please note that any solutions that make use of simulated time will need to be adapted to be truly OpenFL compatible in a real federation by using actual `time.time()` calls (or similar) instead of the simulated time.
# 
# Solutions that use neither **`collaborators_chosen_each_round`** or **`collaborator_times_per_round`** will match the existing OpenFL aggregation customization interface, thus could be used in a real federated deployment using OpenFL.


# the simple example of weighted FedAVG
def weighted_average_aggregation(local_tensors,
                                 tensor_db,
                                 tensor_name,
                                 fl_round,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
    """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        tensor_db: pd.DataFrame that contains global tensors / metrics.
            Columns: ['tensor_name', 'origin', 'round', 'report',  'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    # basic weighted fedavg

    # here are the tensor values themselves
    tensor_values = [t.tensor for t in local_tensors]
    
    # and the weights (i.e. data sizes)
    weight_values = [t.weight for t in local_tensors]
    
    # so we can just use numpy.average
    return np.average(tensor_values, weights=weight_values, axis=0)

# here we will clip outliers by clipping deltas to the Nth percentile (e.g. 80th percentile)
def clipped_aggregation(local_tensors,
                        tensor_db,
                        tensor_name,
                        fl_round,
                        collaborators_chosen_each_round,
                        collaborator_times_per_round):
    """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        tensor_db: pd.DataFrame that contains global tensors / metrics.
            Columns: ['tensor_name', 'origin', 'round', 'report',  'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    # the percentile we will clip to
    clip_to_percentile = 80
    
    # first, we need to determine how much each local update has changed the tensor from the previous value
    # we'll use the tensor_db search function to find the 
    previous_tensor_value = tensor_db.search(tensor_name=tensor_name, fl_round=fl_round, tags=('model',), origin='aggregator')

    if previous_tensor_value.shape[0] > 1:
        print(previous_tensor_value)
        raise ValueError(f'found multiple matching tensors for {tensor_name}, tags=(model,), origin=aggregator')

    if previous_tensor_value.shape[0] < 1:
        # no previous tensor, so just return the weighted average
        return weighted_average_aggregation(local_tensors,
                                            tensor_db,
                                            tensor_name,
                                            fl_round,
                                            collaborators_chosen_each_round,
                                            collaborator_times_per_round)

    previous_tensor_value = previous_tensor_value.nparray.iloc[0]

    # compute the deltas for each collaborator
    deltas = [t.tensor - previous_tensor_value for t in local_tensors]

    # get the target percentile using the absolute values of the deltas
    clip_value = np.percentile(np.abs(deltas), clip_to_percentile)
        
    # let's log what we're clipping to
    logger.info("Clipping tensor {} to value {}".format(tensor_name, clip_value))
    
    # now we can compute our clipped tensors
    clipped_tensors = []
    for delta, t in zip(deltas, local_tensors):
        new_tensor = previous_tensor_value + np.clip(delta, -1 * clip_value, clip_value)
        clipped_tensors.append(new_tensor)
        
    # get an array of weight values for the weighted average
    weights = [t.weight for t in local_tensors]

    # return the weighted average of the clipped tensors
    return np.average(clipped_tensors, weights=weights, axis=0)

# Adapted from FeTS Challenge 2021
# Federated Brain Tumor Segmentation:Multi-Institutional Privacy-Preserving Collaborative Learning
# Ece Isik-Polat, Gorkem Polat,Altan Kocyigit1, and Alptekin Temizel1
def FedAvgM_Selection(local_tensors,
                      tensor_db,
                      tensor_name,
                      fl_round,
                      collaborators_chosen_each_round,
                      collaborator_times_per_round):
    
        """Aggregate tensors.

        Args:
            local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
            tensor_db: Aggregator's TensorDB [writable]. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                - 'round': 0-based number of round corresponding to this tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training result.
                        These tensors are passed to the aggregator node after local learning.
                    - 'aggregated' indicates that tensor is a result of aggregation.
                        These tensors are sent to collaborators for the next round.
                    - 'delta' indicates that value is a difference between rounds
                        for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
        Returns:
            np.ndarray: aggregated tensor
        """
        #momentum
        tensor_db.store(tensor_name='momentum',nparray=0.9,overwrite=False)
        #aggregator_lr
        tensor_db.store(tensor_name='aggregator_lr',nparray=1.0,overwrite=False)

        if fl_round == 0:
            # Just apply FedAvg

            tensor_values = [t.tensor for t in local_tensors]
            weight_values = [t.weight for t in local_tensors]               
            new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        

            #if not (tensor_name in weight_speeds):
            if tensor_name not in tensor_db.search(tags=('weight_speeds',))['tensor_name']:    
                #weight_speeds[tensor_name] = np.zeros_like(local_tensors[0].tensor) # weight_speeds[tensor_name] = np.zeros(local_tensors[0].tensor.shape)
                tensor_db.store(
                    tensor_name=tensor_name, 
                    tags=('weight_speeds',), 
                    nparray=np.zeros_like(local_tensors[0].tensor),
                )
            return new_tensor_weight        
        else:
            if tensor_name.endswith("weight") or tensor_name.endswith("bias"):
                # Calculate aggregator's last value
                previous_tensor_value = None
                for _, record in tensor_db.iterrows():
                    if (record['round'] == fl_round 
                        and record["tensor_name"] == tensor_name
                        and record["tags"] == ("aggregated",)): 
                        previous_tensor_value = record['nparray']
                        break

                if previous_tensor_value is None:
                    logger.warning("Error in fedAvgM: previous_tensor_value is None")
                    logger.warning("Tensor: " + tensor_name)

                    # Just apply FedAvg       
                    tensor_values = [t.tensor for t in local_tensors]
                    weight_values = [t.weight for t in local_tensors]               
                    new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        
                    
                    if tensor_name not in tensor_db.search(tags=('weight_speeds',))['tensor_name']:    
                        tensor_db.store(
                            tensor_name=tensor_name, 
                            tags=('weight_speeds',), 
                            nparray=np.zeros_like(local_tensors[0].tensor),
                        )

                    return new_tensor_weight
                else:
                    # compute the average delta for that layer
                    deltas = [previous_tensor_value - t.tensor for t in local_tensors]
                    weight_values = [t.weight for t in local_tensors]
                    average_deltas = np.average(deltas, weights=weight_values, axis=0) 

                    # V_(t+1) = momentum*V_t + Average_Delta_t
                    tensor_weight_speed = tensor_db.retrieve(
                        tensor_name=tensor_name,
                        tags=('weight_speeds',)
                    )
                    
                    momentum = float(tensor_db.retrieve(tensor_name='momentum'))
                    aggregator_lr = float(tensor_db.retrieve(tensor_name='aggregator_lr'))
                    
                    new_tensor_weight_speed = momentum * tensor_weight_speed + average_deltas # fix delete (1-momentum)
                    
                    tensor_db.store(
                        tensor_name=tensor_name, 
                        tags=('weight_speeds',), 
                        nparray=new_tensor_weight_speed
                    )
                    # W_(t+1) = W_t-lr*V_(t+1)
                    new_tensor_weight = previous_tensor_value - aggregator_lr*new_tensor_weight_speed

                    return new_tensor_weight
            else:
                # Just apply FedAvg       
                tensor_values = [t.tensor for t in local_tensors]
                weight_values = [t.weight for t in local_tensors]               
                new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)

                return new_tensor_weight


# # Running the Experiment
# 
# ```run_challenge_experiment``` is singular interface where your custom methods can be passed.
# 
# - ```aggregation_function```, ```choose_training_collaborators```, and ```training_hyper_parameters_for_round``` correspond to the [this list](#Custom-hyperparameters-for-training) of configurable functions 
# described within this notebook.
# - ```institution_split_csv_filename``` : Describes how the data should be split between all collaborators. Extended documentation about configuring the splits in the ```institution_split_csv_filename``` parameter can be found in the [README.md](https://github.com/FETS-AI/Challenge/blob/main/Task_1/README.md). 
# - ```db_store_rounds``` : This parameter determines how long metrics and weights should be stored by the aggregator before being deleted. Providing a value of `-1` will result in all historical data being retained, but memory usage will likely increase.
# - ```rounds_to_train``` : Defines how many rounds will occur in the experiment
# - ```device``` : Which device to use for training and validation

# ## Setting up the experiment
# Now that we've defined our custom functions, the last thing to do is to configure the experiment. The following cell shows the various settings you can change in your experiment.
# 
# Note that ```rounds_to_train``` can be set as high as you want. However, the experiment will exit once the simulated time value exceeds 1 week of simulated time, or if the specified number of rounds has completed.


# change any of these you wish to your custom functions. You may leave defaults if you wish.
aggregation_function = weighted_average_aggregation
choose_training_collaborators = all_collaborators_train
training_hyper_parameters_for_round = constant_hyper_parameters

# As mentioned in the 'Custom Aggregation Functions' section (above), six 
# perfomance evaluation metrics are included by default for validation outputs in addition 
# to those you specify immediately above. Changing the below value to False will change 
# this fact, excluding the three hausdorff measurements. As hausdorff distance is 
# expensive to compute, excluding them will speed up your experiments.
include_validation_with_hausdorff=False

# We encourage participants to experiment with partitioning_1 and partitioning_2, as well as to create
# other partitionings to test your changes for generalization to multiple partitionings.
#institution_split_csv_filename = 'partitioning_1.csv'
institution_split_csv_filename = 'small_split.csv'

# change this to point to the parent directory of the data
brats_training_data_parent_dir = '/home/ad_tbanda/code/fedAI/MICCAI_FeTS2022_TrainingData'

# increase this if you need a longer history for your algorithms
# decrease this if you need to reduce system RAM consumption
db_store_rounds = 5

# this is passed to PyTorch, so set it accordingly for your system
device = 'cpu'

# you'll want to increase this most likely. You can set it as high as you like, 
# however, the experiment will exit once the simulated time exceeds one week. 
rounds_to_train = 1

# (bool) Determines whether checkpoints should be saved during the experiment. 
# The checkpoints can grow quite large (5-10GB) so only the latest will be saved when this parameter is enabled
save_checkpoints = True

# path to previous checkpoint folder for experiment that was stopped before completion. 
# Checkpoints are stored in checkpoint, and you should provide the experiment directory
# relative to this path (i.e. 'experiment_1'). Please note that if you restore from a checkpoint, 
# and save checkpoint is set to True, then the checkpoint you restore from will be subsequently overwritten.
# restore_from_checkpoint_folder = 'experiment_1'
restore_from_checkpoint_folder = None

# infer participant home folder
home = str(Path.home())

#Creating working directory and copying the required csv files
working_directory= os.path.join(home, '.local/workspace/')
Path(working_directory).mkdir(parents=True, exist_ok=True)
source_dir=f'{Path.cwd()}/openfl-workspace/fets_challenge_workspace/'
pattern = "*.csv"
source_pattern = os.path.join(source_dir, pattern)
files_to_copy = glob.glob(source_pattern)

if not files_to_copy:
    print(f"No files found matching pattern: {pattern}")

for source_file in files_to_copy:
    destination_file = os.path.join(working_directory, os.path.basename(source_file))
    shutil.copy2(source_file, destination_file)
try:
    os.chdir(working_directory)
    print("Directory changed to:", os.getcwd())
except FileNotFoundError:
    print("Error: Directory not found.")
except PermissionError:
    print("Error: Permission denied")

checkpoint_folder = run_challenge_experiment(
    aggregation_function=aggregation_function,
    choose_training_collaborators=choose_training_collaborators,
    training_hyper_parameters_for_round=training_hyper_parameters_for_round,
    include_validation_with_hausdorff=include_validation_with_hausdorff,
    institution_split_csv_filename=institution_split_csv_filename,
    brats_training_data_parent_dir=brats_training_data_parent_dir,
    db_store_rounds=db_store_rounds,
    rounds_to_train=rounds_to_train,
    device=device,
    save_checkpoints=save_checkpoints,
    restore_from_checkpoint_folder = restore_from_checkpoint_folder)


# ## Produce NIfTI files for best model outputs on the validation set
# Now we will produce model outputs to submit to the leader board.
# 
# At the end of every experiment, the best model (according to average ET, TC, WT DICE) 
# is saved to disk at: checkpoint/\<checkpoint folder\>/best_model.pkl,
# where \<checkpoint folder\> is the one printed to stdout during the start of the 
# experiment (look for the log entry: "Created experiment folder experiment_##..." above).

# you will need to specify the correct experiment folder and the parent directory for
# the data you want to run inference over (assumed to be the experiment that just completed)

#checkpoint_folder='experiment_1'
#data_path = </PATH/TO/CHALLENGE_VALIDATION_DATA>
data_path = '/home/ad_tbanda/code/fedAI/MICCAI_FeTS2022_ValidationData'
validation_csv_filename = 'validation.csv'

# you can keep these the same if you wish
final_model_path = os.path.join(working_directory, 'checkpoint', checkpoint_folder, 'best_model.pkl')

# If the experiment is only run for a single round, use the temp model instead
if not Path(final_model_path).exists():
   final_model_path = os.path.join(working_directory, 'checkpoint', checkpoint_folder, 'temp_model.pkl')

outputs_path = os.path.join(working_directory, 'checkpoint', checkpoint_folder, 'model_outputs')

# Using this best model, we can now produce NIfTI files for model outputs
# using a provided data directory

model_outputs_to_disc(data_path=data_path,
                      validation_csv=validation_csv_filename,
                      output_path=outputs_path,
                      native_model_path=final_model_path,
                      outputtag='',
                      device=device)
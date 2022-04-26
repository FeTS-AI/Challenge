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

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # edit according to your system's configuration

from fets_challenge import run_challenge_experiment


# # Adding custom functionality to the experiment
# Within this notebook there are **four** functional areas that you can adjust to improve upon the challenge reference code:
# 
# - [Validation functions](#Custom-Validation-Functions)
# - [Custom aggregation logic](#Custom-Aggregation-Functions)
# - [Selection of training hyperparameters by round](#Custom-hyperparameters-for-training)
# - [Collaborator training selection by round](#Custom-Collaborator-Training-Selection)
# 

# ## Experiment logger for your functions
# The following import allows you to use the same logger used by the experiment framework. This lets you include logging in your functions.

from fets_challenge.experiment import logger

# # Custom Validation Functions

# A list of validation function tuples (string_identifier, function_object) should be provided to the validation_functions argument of  run_challenge_experiment to specify the validation functions to perform in addition to the core performance evaluation metric functions to be run. The string identifier that you included in the 0-index of the tuples providing your additional validation functions will be the assigned name for that metric when it is stored in the aggregator's database. More information about how to use this information for custom aggregation can be found [here](#Using-validation-metrics-for-filtering)
# 
# Default core validation consists of six scores per validation sample: enhancing tumor DICE, tumor core DICE, whole tumor DICE, enhancing tumor hausdorff distance, tumor core hausdorff distance, and whole tumor hausdorff distance. If the parameter to run_challenge_experiment 'include_validation_with_hausdorff' is set to True, only the three DICE scores will be produced as core metrics instead of the full six (this can be done to speed up experiments, as hausdorff is expensive to compute). In order to avoid name collision, we have prepended 'performance_evaluation_metric' to each of the string identifiers used for the core permformance metric functions. 
# 
# Any of the standard PyTorch [validation metrics](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#classification-metrics) can be used to evaluate the model. Any user defined validation functions should conform to the following interface:
# 
#     def validation_fun_interface(targets, predictions):
#         """validation function interface
#     
#         Args:
#             Targets: numpy array of target values
#             Predictions: numpy array of predicted values by the model
#         Returns:
#             val_score : float
#         
#         return val_score
#         
#         
# To add custom metrics to validation that don't conform to the ```(targets, predictions)``` interface, [functool's partial function](https://docs.python.org/3/library/functools.html#functools.partial) can be used to fix a certain number of arguments of a function and generate a new function. For example, we could use [F1 score](https://en.wikipedia.org/wiki/F-score) as a custom metric using the partial function in addition to the exisiting sklearn F1-score metric as follows:
# ```
#     from functools import partial
#     from sklearn.metrics import f1_score
#     validation_functions=[('acc', accuracy), ('f1_score', partial(f1_score, average='macro'))]
# ```
#         
# 
# Sensitivity and Specificity are defined below as reference implementations, each performing an average over the enhancing tumor(ET), tumor core(TC), and whole tumor(WT) regions. We utilize a function that takes the float multi-channel model output and multi-label mask and returns binary outputs and masks for each of ET, TC,and WT.  
# 

from fets_challenge.spec_sens_code import brats_labels


def channel_sensitivity(output, target):
    # computes TP/P for a single channel 

    true_positives = np.sum(output * target)
    total_positives = np.sum(target)

    if total_positives == 0:
        score = 1.0
    else:
        score = true_positives / total_positives
    
    return score


def channel_specificity(output, target):
    # computes TN/N for a single channel

    true_negatives = np.sum((1 - output) * (1 - target))
    total_negatives = np.sum(1 - target)

    if total_negatives == 0:
        score = 1.0
    else:
        score = true_negatives / total_negatives
        
    return score
   
    
def sensitivity(output, target):
    """"
    Calculates the average sensitivity across all of ET, TC, and WT.
    Args:
        Targets: numpy array of target values
        Predictions: numpy array of predicted values by the model
    """        
 
    # parsing model output and target into each of ET, TC, and WT arrays
    brats_val_data = brats_labels(output=output, target=target)
    
    outputs = brats_val_data['outputs']
    targets = brats_val_data['targets']
    
    output_enhancing = outputs['ET'] 
    target_enhancing = targets['ET']

    output_core = outputs['TC'] 
    target_core = targets['TC'] 

    output_whole = outputs['WT'] 
    target_whole = targets['WT']

    sensitivity_for_enhancing = channel_sensitivity(output=output_enhancing, 
                                                    target=target_enhancing)

    sensitivity_for_core = channel_sensitivity(output=output_core, 
                                               target=target_core)

    sensitivity_for_whole = channel_sensitivity(output=output_whole, 
                                                target=target_whole)

    return (sensitivity_for_enhancing + sensitivity_for_core + sensitivity_for_whole) / 3.0
    
    
def specificity(output, target):
    """"
    Calculates the average sensitivity across all of ET, TC, and WT.
    Args:
        Targets: numpy array of target values
        Predictions: numpy array of predicted values by the model
    """  
        
    # parsing model output and target into each of ET, TC, and WT arrays
    brats_val_data = brats_labels(output=output, target=target)
    
    outputs = brats_val_data['outputs']
    targets = brats_val_data['targets']

    
    output_enhancing = outputs['ET'] 
    target_enhancing = targets['ET']

    output_core = outputs['TC'] 
    target_core = targets['TC'] 

    output_whole = outputs['WT'] 
    target_whole = targets['WT']

    specificity_for_enhancing = channel_specificity(output=output_enhancing, 
                                                    target=target_enhancing)

    specificity_for_core = channel_specificity(output=output_core, 
                                               target=target_core)

    specificity_for_whole = channel_specificity(output=output_whole, 
                                                target=target_whole)

    return (specificity_for_enhancing + specificity_for_core + specificity_for_whole) / 3


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
# - **`batches_per_round`**: a flat number of batches each training collaborator will train. Must be an integer or None
# - **`epochs_per_round`**: the number of epochs each training collaborator will train. Must be a float or None. Partial epochs are allowed, such as 0.5 epochs.
# 
# Note that exactly one of **`epochs_per_round`** and **`batches_per_round`** must be `"None"`. You will get an error message and the experiment will terminate if this is not the case to remind you of this requirement.
# 
# Your function will receive the typical aggregator state/history information that it can use to make its determination. The function must return a tuple of (`learning_rate`, `epochs_per_round`, `batches_per_round`). For example, if you return:
# 
# `(1e-4, 2.5, None)`
# 
# then all collaborators selected based on the [collaborator training selection criteria](#Custom-Collaborator-Training-Selection) will train for `2.5` epochs with a learning rate of `1e-4`.
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
        tuple of (learning_rate, epochs_per_round, batches_per_round). One of epochs_per_round and batches_per_round must be None.
    """
    # these are the hyperparameters used in the May 2021 recent training of the actual FeTS Initiative
    # they were tuned using a set of data that UPenn had access to, not on the federation itself
    # they worked pretty well for us, but we think you can do better :)
    epochs_per_round = 1.0
    batches_per_round = None
    learning_rate = 5e-5
    return (learning_rate, epochs_per_round, batches_per_round)


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
        tuple of (learning_rate, epochs_per_round, batches_per_round). One of epochs_per_round and batches_per_round must be None.
    """

    # we'll have a constant learning_rate
    learning_rate = 5e-5
    
    # our epochs per round will start at 1.0 and decay by 0.9 for the first 10 rounds
    epochs_per_round = 1.0
    decay = min(fl_round, 10)
    decay = 0.9 ** decay
    epochs_per_round *= decay    
    
    return (learning_rate, epochs_per_round, None)


# this example has each institution train the same number of batches
def fixed_number_of_batches(collaborators,
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
        tuple of (learning_rate, epochs_per_round, batches_per_round). One of epochs_per_round and batches_per_round must be None.
    """

    # we'll have a constant learning_rate
    learning_rate = 5e-5
    
    # instead of a number of epochs, collaborators will train for a number of batches
    # this means the number of training batches is irrespective of the data sizes at the institutions
    # if the institution has less data than this, they will loop on their data until they have trained
    # the correct number of batches
    batches_per_round = 16
    
    # Note that the middle element (epochs_per_round) is now None
    return (learning_rate, None, batches_per_round)


# # Custom Aggregation Functions
# Standard aggregation methods allow for simple layer-wise combination (via weighted_mean, mean, median, etc.); however, more complex aggregation methods can be supported by evaluating collaborator metrics, weights from prior rounds, etc. OpenFL enables custom aggregation functions via the [**AggregationFunctionInterface**](https://github.com/intel/openfl/blob/fets/openfl/component/aggregation_functions/interface.py). For the challenge, we wrap this interface so we can pass additional simulation state, such as simulated time.
# 
# [**LocalTensors**](https://github.com/intel/openfl/blob/fets/openfl/utilities/types.py#L13) are named tuples of the form `('collaborator_name', 'tensor', 'collaborator_weight')`. Your custom aggregation function will be passed a list of LocalTensors, which will contain an entry for each collaborator who participated in the prior training round. The [**`tensor_db`**](#Getting-access-to-historical-weights,-metrics,-and-more) gives direct access to the aggregator's tensor_db dataframe and includes all tensors / metrics reported by collaborators. Using the passed tensor_db reference, participants may even store custom information by using in-place write operations. A few examples are included below.
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
    # we'll use the db_iterator to find the previous round's value for this tensor
    previous_tensor_value = None
    for _, record in tensor_db.iterrows():
        if (
            record['round'] == (fl_round - 1)
            and record['tensor_name'] == tensor_name
            and 'aggregated' in record['tags']
            and 'delta' not in record['tags']
           ):
            previous_tensor_value = record['nparray']
            break
       
    # if we have no previous tensor_value, we won't actually clip
    if previous_tensor_value is None:
        clipped_tensors = [t.tensor for t in local_tensors]
    # otherwise, we will use clipped deltas
    else:
        # compute the deltas
        deltas = [t.tensor - previous_tensor_value for t in local_tensors]
    
        # concatenate all the deltas
        all_deltas = np.concatenate(deltas)
        
        # take the absolute value
        abs_deltas = np.abs(all_deltas)
        
        # finally, get the 80th percentile
        clip_value = np.percentile(abs_deltas, clip_to_percentile)
        
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


# # Running the Experiment
# 
# ```run_challenge_experiment``` is singular interface where your custom methods can be passed.
# 
# - ```aggregation_function```, ```choose_training_collaborators```, ```training_hyper_parameters_for_round```, and ```validation_functions``` correspond to the [this list](#Custom-hyperparameters-for-training) of configurable functions 
# described within this notebook.
# - ```validation_functions``` should be a list of tuples, enabling you add multiple additional validation functions. The tuples should be `(name, function)`, where `name` is the string that will be associated with the metric in the `tensor_db`, and `function` is the python function you implemented above. It can be an empty list if you do not wish to add additional validation functions.
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
validation_functions = [('sensitivity', sensitivity), ('specificity', specificity)]

# As mentioned in the 'Custom Aggregation Functions' section (above), six 
# perfomance evaluation metrics are included by default for validation outputs in addition 
# to those you specify immediately above. Changing the below value to False will change 
# this fact, excluding the three hausdorff measurements. As hausdorff distance is 
# expensive to compute, excluding them will speed up your experiments.
include_validation_with_hausdorff=True

# We encourage participants to experiment with partitioning_1 and partitioning_2, as well as to create
# other partitionings to test your changes for generalization to multiple partitionings.
institution_split_csv_filename = 'partitioning_1.csv'

# change this to point to the parent directory of the data
brats_training_data_parent_dir = '/raid/datasets/FeTS21/MICCAI_FeTS2021_TrainingData'

# increase this if you need a longer history for your algorithms
# decrease this if you need to reduce system RAM consumption
db_store_rounds = 5

# this is passed to PyTorch, so set it accordingly for your system
device = 'cuda'

# you'll want to increase this most likely. You can set it as high as you like, 
# however, the experiment will exit once the simulated time exceeds one week. 
rounds_to_train = 5

# challenge_metrics_validation_interval is parameter that determines how often the
# challenge metrics should be computed. Some of the metrics, like Hausdorff distance,
# take a long time to compute, so increasing this value will speed up your total training
# quite significantly. The default is to compute challenge metrics every other round (round_num % 2)
challenge_metrics_validation_interval = 2

# (bool) Determines whether checkpoints should be saved during the experiment. 
# The checkpoints can grow quite large (5-10GB) so only the latest will be saved when this parameter is enabled
save_checkpoints = True

# path to previous checkpoint folder for experiment that was stopped before completion. 
# Checkpoints are stored in ~/.local/workspace/checkpoint, and you should provide the experiment directory 
# relative to this path (i.e. 'experiment_1'). Please note that if you restore from a checkpoint, 
# and save checkpoint is set to True, then the checkpoint you restore from will be subsequently overwritten.
# restore_from_checkpoint_folder = 'experiment_1'
restore_from_checkpoint_folder = None


# the scores are returned in a Pandas dataframe
scores_dataframe = run_challenge_experiment(
    aggregation_function=aggregation_function,
    choose_training_collaborators=choose_training_collaborators,
    training_hyper_parameters_for_round=training_hyper_parameters_for_round,
    validation_functions=validation_functions,
    include_validation_with_hausdorff=include_validation_with_hausdorff,
    institution_split_csv_filename=institution_split_csv_filename,
    brats_training_data_parent_dir=brats_training_data_parent_dir,
    db_store_rounds=db_store_rounds,
    rounds_to_train=rounds_to_train,
    device=device,
    challenge_metrics_validation_interval=challenge_metrics_validation_interval,
    save_checkpoints=save_checkpoints,
    restore_from_checkpoint_folder = restore_from_checkpoint_folder)


scores_dataframe


# ## Produce NIfTI files for best model outputs on the validation set
# Now we will produce model outputs to submit to the leader board.
# 
# At the end of every experiment, the best model (according to average ET, TC, WT DICE) 
# is saved to disk at: ~/.local/workspace/checkpoint/\<checkpoint folder\>/best_model.pkl,
# where \<checkpoint folder\> is the one printed to stdout during the start of the 
# experiment (look for the log entry: "Created experiment folder experiment_##..." above).


from fets_challenge import model_outputs_to_disc
from pathlib import Path

# infer participant home folder
home = str(Path.home())

# you will need to specify the correct experiment folder and the parent directory for
# the data you want to run inference over
checkpoint_folder='experiment_1'
data_path = </PATH/TO/CHALLENGE_VALIDATION_DATA>

# you can keep these the same if you wish
best_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'best_model.pkl')
outputs_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'model_outputs')


# Using this best model, we can now produce NIfTI files for model outputs 
# using a provided data directory

model_outputs_to_disc(data_path=data_path, 
                      output_path=outputs_path, 
                      native_model_path=best_model_path,
                      outputtag='',
                      device=device)

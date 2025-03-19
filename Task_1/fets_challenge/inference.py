# Provided by the FeTS Initiative (www.fets.ai) as part of the FeTS Challenge 2022

# Contributing Authors (alphabetical):
# Brandon Edwards (Intel), Patrick Foley (Intel), Micah Sheller (Intel)

import os
from copy import copy
from logging import getLogger
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import torch, torchio

import openfl.native as fx
from .gandlf_csv_adapter import construct_fedsim_csv
from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.config_manager import ConfigManager
from .fets_challenge_model import FeTSChallengeModel

logger = getLogger(__name__)

# some hard coded variables
channel_keys = ['1', '2', '3', '4']
class_label_map = {0:0, 1:1, 2:2, 4:4}
class_list = list(np.sort(list(class_label_map.values())))

def nan_check(tensor, tensor_description):
    tensor = tensor.cpu()
    if torch.any(torch.isnan(tensor)):
        raise ValueError("A " + tensor_description + " was found to have nan values.")


def binarize(array, threshold=0.5):
    """
    Get binarized output using threshold. 
    """
    
    if (np.amax(array) > 1.0) or (np.amin(array) < 0.0):
        raise ValueError('Voxel value fed to lambda in converting to original labels was out of range.')
    
    # obtain binarized output
    binarized = array.copy()
    zero_mask = (binarized <= threshold)
    binarized[zero_mask] = 0.0
    binarized[~zero_mask] = 1.0
    
    return binarized


def get_binarized_and_belief(array, threshold=0.5):
    
    """
    Get binarized output using threshold and report the belief values for each of the
    channels along the last axis. Belief value is a list of indices along the last axis, and is 
    determined by the order of how close (closest first) each binarization is from its original in this last axis
    so belief should have the same shape as arrray.
    Assumptions:
    - array is float valued in the range [0.0, 1.0]
    """
    
    # check assumption above
    if (np.amax(array) > 1.0) or (np.amin(array) < 0.0):
        raise ValueError('Voxel value fed to lambda in converting to original labels was out of range.')
        
    # obtain binarized output
    binarized = binarize(array=array, threshold=threshold)
    
    # we will sort from least to greatest, so least suspicion is what we will believe
    raw_suspicion = np.absolute(array - binarized)
    
    belief = np.argsort(raw_suspicion, axis=-1)
    
    return binarized, belief

def generate_validation_csv(data_path, validation_csv_filename, working_dir):
    """
    Create the validation CSV to be consumed by the FeTSChallengeTaskRunner
    """
    validation_csv_path = os.path.join(working_dir, validation_csv_filename)
    validation_csv_dict = construct_fedsim_csv(data_path,
                              validation_csv_path,
                              0.0,
                              'placeholder',
                              training_and_validation=False)
    validation_csv_dict.to_csv(os.path.join(working_dir, 'validation_paths.csv'),index=False)

def replace_initializations(done_replacing, array, mask, replacement_value, initialization_value):
    """
    Replace in array[mask] intitialization values with replacement value, 
    ensuring that the locations to replace all held initialization values
    """
    
    # sanity check
    if np.any(mask) and done_replacing:
        raise ValueError('Being given locations to replace and yet told that we are done replacing.')
        
    # check that the mask and array have the same shape
    if array.shape != mask.shape:
        raise ValueError('Attempting to replace using a mask shape: {} not equal to the array shape: {}'.format(mask.shape, array.shape))
    
    # check that the mask only points to locations with initialized values
    if np.any(array[mask] != initialization_value):
        raise ValueError('Attempting to overwrite a non-initialization value.')
        
    array[mask] = replacement_value
    
    done_replacing = np.all(array!=initialization_value)
    
    return array, done_replacing


def check_subarray(array1, array2):
    """
    Checks to see where array2 is a subarray of array1.
    Assumptions:
    - array2 has one axis and is equal in length to the last axis of array1
    """
    
    # check assumption
    if (len(array2.shape) != 1) or (array2.shape[0] != array1.shape[-1]):
        raise ValueError('Attempting to check for subarray equality when shape assumption does not hold.')
        
    return np.all(array1==array2, axis=-1)


def convert_to_original_labels(array, threshold=0.5, initialization_value=999):
    """
    array has float output in the range [0.0, 1.0]. 
    Last three channels are expected to correspond to ET, TC, and WT respecively.
    
    """
    
    binarized, belief = get_binarized_and_belief(array=array, threshold=threshold)
    
    #sanity check
    if binarized.shape != belief.shape:
        raise ValueError('Sanity check did not pass.')
        
    # initialize with a crazy label we will be sure is gone in the end
    slice_all_but_last_channel = tuple([slice(None) for _ in array.shape[:-1]] + [0])
    original_labels = initialization_value * np.ones_like(array[slice_all_but_last_channel])
    
    # the outer keys correspond to the binarized values
    # the inner keys correspond to the order of indices comingn from argsort(ascending) on suspicion, i.e. 
    # how far the binarized sigmoid outputs were from the original sigmoid outputs 
    #     for example, (2, 1, 0) means the suspicion from least to greatest was: 'WT', 'TC', 'ET'
    #     (recall that the order of the last three channels is expected to be: 'ET', 'TC', and 'WT')
    mapper = {(0, 0, 0): 0, 
              (1, 1, 1): 4,
              (0, 1, 1): 1,
              (0, 0, 1): 2,
              (0, 1, 0): {(2, 0, 1): 0,
                          (2, 1, 0): 0, 
                          (1, 0, 2): 1,
                          (1, 2, 0): 1,
                          (0, 2, 1): 0,
                          (0, 1, 2): 1}, 
              (1, 1, 0): {(2, 0, 1): 0,
                          (2, 1, 0): 0, 
                          (1, 0, 2): 4,
                          (1, 2, 0): 4,
                          (0, 2, 1): 4,
                          (0, 1, 2): 4},
              (1, 0, 1): {(2, 0, 1): 4,
                          (2, 1, 0): 2, 
                          (1, 0, 2): 2,
                          (1, 2, 0): 2,
                          (0, 2, 1): 4,
                          (0, 1, 2): 4}, 
              (1, 0, 0): {(2, 0, 1): 0,
                          (2, 1, 0): 0, 
                          (1, 0, 2): 0,
                          (1, 2, 0): 0,
                          (0, 2, 1): 4,
                          (0, 1, 2): 4}}
    
    
    
    done_replacing = False
    
    for binary_key, inner in mapper.items():
        mask1 = check_subarray(array1=binarized, array2=np.array(binary_key))
        if isinstance(inner, int):
            original_labels, done_replacing = replace_initializations(done_replacing=done_replacing, 
                                                                      array=original_labels, 
                                                                      mask=mask1, 
                                                                      replacement_value=inner, 
                                                                      initialization_value=initialization_value)
        else:
            for inner_key, inner_value in inner.items():
                mask2 = np.logical_and(mask1, check_subarray(array1=belief, array2=np.array(inner_key)))
                original_labels, done_replacing = replace_initializations(done_replacing=done_replacing,
                                                                          array=original_labels, 
                                                                          mask=mask2, 
                                                                          replacement_value=inner_value, 
                                                                          initialization_value=initialization_value)
        
    if not done_replacing:
        raise ValueError('About to return so should have been done replacing but told otherwise.')
        
    return original_labels.astype(np.uint8)


def model_outputs_to_disc(data_path, 
                          validation_csv,
                          output_path, 
                          native_model_path,
                          outputtag='',
                          device='cpu'):
    
    from sys import path, exit
    
    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()

    generate_validation_csv(data_path,validation_csv, working_dir=work)
    
    # # overwrite datapath value for a single 'InferenceCol' collaborator
    # plan.cols_data_paths['InferenceCol'] = data_path
    
    # # get the inference data loader
    # data_loader = copy(plan).get_data_loader('InferenceCol')

    # # get the task runner, passing the data loader
    # task_runner = copy(plan).get_task_runner(data_loader)
    gandlf_config_path = os.path.join(root, 'config', 'gandlf_config.yaml')
    fets_model = FeTSChallengeModel(gandlf_config_path)
    val_csv_path = os.path.join(work, 'validation_paths.csv')
    gandlf_conf = ConfigManager(gandlf_config_path)
    (
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        params,
    ) = create_pytorch_objects(
        gandlf_conf, train_csv=None, val_csv=val_csv_path, device=device
    )
    gandlf_conf['output_dir'] = output_path
    gandlf_conf['save_output'] = True
    fets_model.model = model
    fets_model.optimizer = optimizer
    fets_model.scheduler = scheduler
    fets_model.params = params
    fets_model.device = device

    # Populate model weights
    device = torch.device(device)
    fets_model.load_native(filepath=native_model_path, map_location=device)
    #task_runner.opt_treatment = 'RESET'
    
    logger.info('Starting inference using data from {}\n'.format(data_path))
    
    fets_model.inference('aggregator',-1,val_loader,apply='global')
    logger.info(f"\nFinished generating predictions to output folder {output_path}")

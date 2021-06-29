# Provided by the FeTS Initiative (www.fets.ai) as part of the FeTS Challenge 2021

# Contributing Authors (alphabetical):
# Brandon Edwars (Intel), Patrick Foley (Intel), Micah Sheller (Intel)

import os
from copy import copy
from logging import getLogger
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import torch, torchio

import openfl.native as fx

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
                          output_path, 
                          native_model_path,
                          outputtag='',
                          device='cpu'):

    fx.init('fets_challenge_workspace')
    
    from sys import path, exit
    
    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()

    path.append(str(root))
    path.insert(0, str(work))
    
    overrides = {
        'task_runner.settings.device': device,
        'data_loader.settings.data_usage': 'inference',
        'data_loader.settings.federated_simulation_institution_name': None,
    }
    
    # Update the plan if necessary
    plan = fx.update_plan(overrides)

    # overwrite datapath value for a single 'InferenceCol' collaborator
    plan.cols_data_paths['InferenceCol'] = data_path
    
    # get the inference data loader
    data_loader = copy(plan).get_data_loader('InferenceCol')

    # get the task runner, passing the data loader
    task_runner = copy(plan).get_task_runner(data_loader)
    
    # Populate model weights
    device = torch.device(device)
    task_runner.load_native(filepath=native_model_path, map_location=device)
    task_runner.to(device)

    
    
    logger.info('Starting inference using data from {}\n'.format(data_path))
    
    inference_loader = data_loader.get_inference_loader()
    
    for subject in inference_loader:
        
        subfolder = subject['subject_id'][0]
        
        #prep the path for the output file
        if not os.path.exists(output_path):
            os.mkdir(output_path)
         
        inference_outpath = os.path.join(output_path, subfolder + outputtag + '_seg.nii.gz')
        
        logger.info("Validating with subject: {}".format(subfolder))
        features = torch.cat([subject[key][torchio.DATA] for key in channel_keys], dim=1).float()
        nan_check(tensor=features, tensor_description='features tensor')
          
        task_runner.sanity_check_val_input_shape(features)
        output = task_runner.data.infer_with_patches(model_inference_function=[task_runner.infer_batch_with_no_numpy_conversion], 
                                                              features=features)
        nan_check(tensor=output, tensor_description='model output tensor')
        task_runner.sanity_check_val_output_shape(output)
          
        output = np.squeeze(output.cpu().numpy())

        # GANDLFData loader produces transposed output from what sitk gets from file, so transposing here.
        output = np.transpose(output)

        # process float sigmoid outputs (three channel corresponding to ET, TC, and WT)
        # into original label output (no channels, but values in 0, 1, 2, 4)
        output = convert_to_original_labels(output)
  
        # convert array to SimpleITK image 
        image = sitk.GetImageFromArray(output)

        # get the image information such as affine orientation from the label
        image.CopyInformation(sitk.ReadImage(subject['label']['path']))

        logger.info("\nWriting inference NIfTI image of shape {} to {}".format(output.shape, inference_outpath))
        sitk.WriteImage(image, inference_outpath)
        
        
                          
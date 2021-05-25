# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# The following code is modified from https://github.com/CBICA/BrainMaGe which has the following license:

# Copyright 2020 Center for Biomedical Image Computing and Analytics, University of Pennsylvania
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# This is a 3-clause BSD license as defined in https://opensource.org/licenses/BSD-3-Clause

# Provided by the FeTS Initiative (www.fets.ai) as part of the FeTS Challenge 2021

# Contributing Authors (alphabetical):
# Brandon Edwards (Intel)
# Sarthak Pati (University of Pennsylvania)
# Micah Sheller (Intel)


import os
os.environ['TORCHIO_HIDE_CITATION_PROMPT'] = '1' # hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
import torchio

import logging
import numpy as np
import time
import sys, os
import ast
import tqdm
from math import ceil
from itertools import product

import pandas as pd
import random
from copy import deepcopy

import torchio
import torch
import torch.optim as optim
from torch.autograd import Variable

from GANDLF.utils import one_hot

from openfl.federated.task import PyTorchTaskRunner
from openfl.utilities import TensorKey, split_tensor_dict_for_holdouts

from fets.models.pytorch.brainmage.losses import MCD_loss, MCD_MSE_loss, dice_loss
from fets.models.pytorch.brainmage.losses import brats_dice_loss, brats_dice_log_loss, brats_dice, brats_dice_loss_w_background, brats_dice_loss_w_crossentropy
from fets.models.pytorch.brainmage.losses import background_dice_loss, crossentropy, dice_loss_skipping_first_channel, dice_loss_all_channels, mirrored_brats_dice_loss
from fets.models.pytorch.brainmage.losses import fets_phase2_validation

from fets.models.pytorch.brainmage.seg_modules import in_conv, DownsamplingModule, EncodingModule, InceptionModule, ResNetModule
from fets.models.pytorch.brainmage.seg_modules import UpsamplingModule, DecodingModule,IncDownsamplingModule,IncConv
from fets.models.pytorch.brainmage.seg_modules import out_conv, FCNUpsamplingModule, IncDropout,IncUpsamplingModule

from . import TRAINING_HPARAMS

 # TODO: Temporarily using patching in the model code (until GANDLF patching is plugged in) 
def random_slices(array, psize):
    # an example expected shape is: (1, 1, 240, 240, 155)
    # the patch will not apply to the first two dimensions
    shape = array.shape[2:]
    slices = [slice(None), slice(None)]
    for axis, length in enumerate(psize):
        if shape[axis] > length:
            shift = random.randint(0,shape[axis] - length)
            slices.append(slice(shift, shift + length))
    return slices


def crop(array, slices):
    return array[tuple(slices)]


def nan_check(tensor, tensor_description):
    tensor = tensor.cpu()
    if torch.any(torch.isnan(tensor)):
        raise ValueError("A " + tensor_description + " was found to have nan values.")

# this is an adaptation of fets.models.pytorch.brainmage.BrainMage 3dresunet
class FeTSChallengeModel(PyTorchTaskRunner):

    def __init__(self,
                 data_loader,
                 final_layer_activation=None, 
                 sigmoid_input_multiplier=1.0,
                 val_input_shape=None,
                 val_output_shape=None,
                 validation_functions=[],
                 **kwargs):

        # adapting IL openfl interface to IOTG openfl interface
        data = data_loader
        self.data = data

        if val_input_shape is None:
            val_input_shape = [data.batch_size, 4, 240, 240, 155]
        if val_output_shape is None:
            val_output_shape = [data.batch_size, len(data.class_list), 240, 240, 155]
        
        # I've merged a sub and super class together here
        self.__init_2(val_input_shape=val_input_shape, val_output_shape=val_output_shape, data=data, **kwargs)

        if final_layer_activation is None:
            # inferring from data object class_list attribute
            if (self.data.class_list == [0, 1]) or (self.data.class_list == ['4', '1||4', '1||2||4']) or (self.data.class_list == ['4', '1||4']):
                # single output channel or multi-label
                final_layer_activation = 'sigmoid'
            elif self.data.class_list == [0, 1, 2, 4]:
                # mutually exclusive labels
                final_layer_activation = 'softmax'
            else:
                raise ValueError('No final_layer_activation provided and not able to infer the value needed.')      

        self.init_network(device=self.device, 
                          final_layer_activation=final_layer_activation, 
                          sigmoid_input_multiplier=sigmoid_input_multiplier)
        self._init_optimizer()
        
        self.initialize_tensorkeys_for_functions()
        self.validation_functions = validation_functions
        

    def init_network(self, device, print_model=False, final_layer_activation='softmax', sigmoid_input_multiplier=1.0, **kwargs):
        self.ins = in_conv(self.n_channels, self.base_filters, res=True)
        self.ds_0 = DownsamplingModule(self.base_filters, self.base_filters*2)
        self.en_1 = EncodingModule(self.base_filters*2, self.base_filters*2, res=True)
        self.ds_1 = DownsamplingModule(self.base_filters*2, self.base_filters*4)
        self.en_2 = EncodingModule(self.base_filters*4, self.base_filters*4, res=True)
        self.ds_2 = DownsamplingModule(self.base_filters*4, self.base_filters*8)
        self.en_3 = EncodingModule(self.base_filters*8, self.base_filters*8, res=True)
        self.ds_3 = DownsamplingModule(self.base_filters*8, self.base_filters*16)
        self.en_4 = EncodingModule(self.base_filters*16, self.base_filters*16, res=True)
        self.us_3 = UpsamplingModule(self.base_filters*16, self.base_filters*8)
        self.de_3 = DecodingModule(self.base_filters*16, self.base_filters*8, res=True)
        self.us_2 = UpsamplingModule(self.base_filters*8, self.base_filters*4)
        self.de_2 = DecodingModule(self.base_filters*8, self.base_filters*4, res=True)
        self.us_1 = UpsamplingModule(self.base_filters*4, self.base_filters*2)
        self.de_1 = DecodingModule(self.base_filters*4, self.base_filters*2, res=True)
        self.us_0 = UpsamplingModule(self.base_filters*2, self.base_filters)
        self.out = out_conv(self.base_filters*2, 
                            self.label_channels, 
                            res=True, 
                            activation=final_layer_activation, 
                            sigmoid_input_multiplier=sigmoid_input_multiplier)

        if print_model:
            print(self)

        # send this to the device
        self.to(device)

    def forward(self, x):
        # normalize input if can do so without producing nan values

        if (torch.isnan(torch.std(x)).cpu().item() != True) and (torch.std(x).cpu().item() != 0.0):
            x = (x - torch.mean(x)) / torch.std(x)
        else:
            self.logger.debug("Skipping input normalization due to std val of: {}.".format(torch.std(x).cpu().item()))
        x1 = self.ins(x)
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        x = self.out(x, x1)
        return x


    def __init_2(self, 
                 data,
                 base_filters, 
                 loss_function,
                 validation_function, 
                 opt,
                 lr,
                 device='cpu',
                 n_classes=4,
                 n_channels=4,
                 psize=[128,128,128],
                 smooth=1e-7,
                 use_penalties=False, 
                 validate_without_patches = False,
                 validate_with_fine_grained_dice = True, 
                 torch_threads=None, 
                 kmp_affinity=False, 
                 loss_function_kwargs={}, 
                 validation_function_kwargs={},
                 val_input_shape = None,
                 val_output_shape = None,
                 **kwargs):
        super().__init__(data_loader=data, device=device, **kwargs)

        self.logger = logging.getLogger()

        if torch_threads is not None:
            torch.set_num_threads(torch_threads)
        if kmp_affinity:
            os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
                 
        self.device = device

        # FIXME: this puts priority for these values on data object over flplan. Is this correct?
        if hasattr(data, 'n_classes') and data.n_classes is not None:
            self.n_classes = data.n_classes
        else:
            self.n_classes = n_classes

        if hasattr(data, 'n_channels') and data.n_channels is not None:
            self.n_channels = data.n_channels
        else:
            self.n_channels = n_channels

        if hasattr(data, 'psize') and data.psize is not None:
            self.psize = data.psize
        else:
            self.psize = psize

        self.lr = lr

        self.which_loss = loss_function
        self.which_validation = validation_function
        self.opt = opt
        #TODO: Binary classficition with one channel is currently not supported
        self.label_channels = self.n_classes
        self.base_filters = base_filters
        self.smooth = smooth
        self.which_model = self.__repr__()
        self.use_panalties = use_penalties

        self.loss_function_kwargs = loss_function_kwargs
        self.validation_function_kwargs = validation_function_kwargs

        # used only when using the gandlf_data object
        # (will we crop external zero-planes, infer, then pad output with zeros OR
        #  get outputs for multiple patches - fusing the outputs)
        self.validate_without_patches = validate_without_patches

        # Determines if we want our validation results to include separate values for whole-tumor, tumor-core, and
        # enhancing tumor, or to simply report the average of those
        self.validate_with_fine_grained_dice = validate_with_fine_grained_dice

        # if not None, used to sanity check what input and output shapes are for validation pipeline 
        self.val_input_shape = val_input_shape
        self.val_output_shape = val_output_shape
        
        ############### CHOOSING THE LOSS AND VALIDATION FUNCTIONS ###################

        # hard coded for now
        #FIXME: Note dependency on this and loss_function_kwargs on total_valscore definition in validate method
        # I try to track this with self.validation_output_keys (below)
        if self.which_validation == 'fets_phase2_validation':
            self.validation_function = fets_phase2_validation
            self.validation_output_keys = ['binary_DICE_ET', 
                                           'binary_DICE_TC', 
                                           'binary_DICE_WT', 
                                           'binary_Hausdorff95_ET', 
                                           'binary_Hausdorff95_TC', 
                                           'binary_Hausdorff95_WT']
        else:
            raise ValueError('The validation function {} is not currently supported'.format(self.which_validation))

        # old dc is now dice_loss_skipping_first_channel
        if self.which_loss == 'brats_dice_loss':
            self.loss_fn = brats_dice_loss
        elif self.which_loss == 'brats_dice_log_loss':
            self.loss_fn = brats_dice_log_loss
        elif self.which_loss == 'brats_dice_loss_w_background':
            self.loss_fn = brats_dice_loss_w_background
        elif self.which_loss == 'brats_dice_loss_w_crossentropy':
            self.loss_fn = brats_dice_loss_w_crossentropy
        elif self.which_loss == 'crossentropy':
            self.loss_fn = crossentropy
        elif self.which_loss == 'background_dice_loss':
            self.loss_fn = background_dice_loss
        elif self.which_loss == 'dice_loss_skipping_first_channel':
            self.loss_fn = dice_loss_skipping_first_channel
        elif self.which_loss == 'dice_loss_all_channels':
            self.loss_fn = dice_loss_all_channels
        elif self.which_loss == 'mirrored_brats_dice_loss':
            self.loss_fn = mirrored_brats_dice_loss
        else:
            raise ValueError('{} loss is not supported'.format(self.which_loss))

        self.channel_keys= self.get_channel_keys()
        
        self.dice_penalty_dict = None
        if self.use_panalties:
            # prepare penalties dict
            _, self.dice_penalty_dict = self.prep_penalties()

    def _init_optimizer(self):
        if self.opt == 'sgd':
            self.optimizer = optim.SGD(self.parameters(),
                                       lr= self.lr,
                                       momentum = 0.9)
        if self.opt == 'adam':    
            self.optimizer = optim.Adam(self.parameters(), 
                                        lr = self.lr, 
                                        betas = (0.9,0.999), 
                                        weight_decay = 0.00005)

    def reset_opt_vars(self, **kwargs):
        self._init_optimizer()

    def get_channel_keys(self):
        # Getting one training subject
        channel_keys = []
        for subject in self.data.get_train_loader():
            # break
          
            # use last subject to inspect channel keys
            # channel_keys = []
            for key in subject.keys():
                if key.isnumeric():
                    channel_keys.append(key)

            return channel_keys
        return channel_keys

    def sanity_check_val_input_shape(self, features):
        features_shape = list(features.shape)
        if (self.val_input_shape is not None) and (self.val_input_shape != features_shape):
            # FIXME: (replace with raised exception?)
            self.logger.debug('Features going into model during validation has shape {} when {} was expected.'.format(features_shape, self.val_input_shape))

    def sanity_check_val_output_shape(self, output):
        output_shape = list(output.shape)
        if (self.val_output_shape is not None) and (self.val_output_shape != output_shape):
            # FIXME: (replace with raised exception?)
            self.logger.debug('Output from the model during validation has shape {} when {} was expected.'.format(output_shape, self.val_output_shape))

    def prep_penalties(self):

        # initialize without considering background
        dice_weights_dict = {} # average for "weighted averaging"
        dice_penalty_dict = {} # penalty for misclassification
        for i in range(1, self.n_classes):
            dice_weights_dict[i] = 0
            dice_penalty_dict[i] = 0

        penalty_loader = self.data.get_penalty_loader()
        
        # get the weights for use for dice loss
        total_nonZeroVoxels = 0
        
        # dice penalty is calculated on the basis of the masks (processed here) and predicted labels
        # iterate through full data (may differ from training data by not being cropped for example)
        for subject in penalty_loader: 
            # accumulate dice weights for each label
            mask = subject['label'][torchio.DATA]
            one_hot_mask = one_hot(mask, self.data.class_list)
            for i in range(0, self.n_classes):
                currentNumber = torch.nonzero(one_hot_mask[:,i,:,:,:], as_tuple=False).size(0)
                dice_weights_dict[i] = dice_weights_dict[i] + currentNumber # class-specific non-zero voxels
                total_nonZeroVoxels = total_nonZeroVoxels + currentNumber # total number of non-zero voxels to be considered
        
        if total_nonZeroVoxels == 0:
            raise RuntimeError('Trying to train on data where every label mask is background class only.')

        # dice_weights_dict_temp = deepcopy(dice_weights_dict)
        dice_weights_dict = {k: (v / total_nonZeroVoxels) for k, v in dice_weights_dict.items()} # divide each dice value by total nonzero
        dice_penalty_dict = deepcopy(dice_weights_dict) # deep copy so that both values are preserved
        dice_penalty_dict = {k: 1 - v for k, v in dice_weights_dict.items()} # subtract from 1 for penalty
        total = sum(dice_penalty_dict.values())
        dice_penalty_dict = {k: v / total for k, v in dice_penalty_dict.items()} # normalize penalty to ensure sum of 1
        # dice_penalty_dict = get_class_imbalance_weights(trainingDataFromPickle, parameters, headers, is_regression, class_list) # this doesn't work because ImagesFromDataFrame gets import twice, causing a "'module' object is not callable" error

        return dice_weights_dict, dice_penalty_dict

    def infer_batch_with_no_numpy_conversion(self, features, **kwargs):
        """Very similar to base model infer_batch, but does not
           explicitly convert the output to numpy.
           Run inference on a batch
        Args:
            features: Input for batch
        Gets the outputs for the inputs provided.
        """

        device = torch.device(self.device)
        self.eval()

        with torch.no_grad():
            features = features.to(device)
            output = self(features)
            output = output.cpu()
        return output

    def train_batches(self, col_name, round_num, input_tensor_dict, use_tqdm=False, **kwargs):

        # handle the hparams
        epochs_per_round = float(input_tensor_dict.pop('epochs_per_round'))
        num_batches = int(input_tensor_dict.pop('batches_per_round'))
        learning_rate = float(input_tensor_dict.pop('learning_rate'))

        # determine the number of batches
        # if num_batches is less than 1, we use epochs to compute the number of batches
        if num_batches < 1:
            num_batches = ceil(epochs_per_round * len(self.data.train_loader.dataset))

        # set the learning rate
        self.lr = learning_rate

        # rebuild the model
        self.rebuild_model(round_num, input_tensor_dict)
        
        device = torch.device(self.device)

        ################################ LOGGING SOME STUFF ######################
        self.logger.debug("Hostname   : {}".format(str(os.getenv("HOSTNAME"))))
        sys.stdout.flush()

        self.logger.debug("Training batches: {}".format(len(self.data.train_loader.dataset)))
        sys.stdout.flush()

        self.logger.debug('Using device: {}'.format(device))
        if device.type == 'cuda':
            self.logger.debug('Memory Allocated: {}GB'.format(round(torch.cuda.memory_allocated(0)/1024**3, 1)))
            self.logger.debug('Memory Cached: {}GB'.format(round(torch.cuda.memory_cached(0)/1024**3, 1)))

        sys.stdout.flush()

        train_loader = self.data.get_train_loader()

        if train_loader == []:
            raise RuntimeError("Attempting to run training with an empty training loader.")

        if use_tqdm:
            train_loader = tqdm.tqdm(train_loader, desc="training for this round")

        total_loss = 0
        batch_num = 0
        
        # set to "training" mode
        self.train()
        while batch_num < num_batches:
                       
            for batch in train_loader:
                if batch_num >= num_batches:
                    break
                else:
                    # Load the batch and its ground truth
                    
                    # this is when we are using pt_brainmagedata
                    if ('features' in batch.keys()) and ('gt' in batch.keys()):
                        features = batch['features']
                        nan_check(tensor=features, tensor_description='features tensor')
                        mask = batch['gt']
                        nan_check(tensor=mask, tensor_description='ground truth mask tensor')
                    # this is when we are using gandlf loader   
                    else:
                        self.logger.debug("Training on batch with subjects: {}".format(batch['subject_id']))
                        features = torch.cat([batch[key][torchio.DATA] for key in self.channel_keys], dim=1).float()
                        nan_check(tensor=features, tensor_description='features tensor')
                        mask = batch['label'][torchio.DATA]
                        nan_check(tensor=mask, tensor_description='ground truth mask tensor')

                    mask = one_hot(mask, self.data.class_list).float()
                    nan_check(tensor=mask, tensor_description='one_hot ground truth mask tensor')
                        
                    # Loading features into device
                    features, mask = features.to(device), mask.to(device)
                    
                    # Making sure that the optimizer has been reset
                    self.optimizer.zero_grad()

                    # Forward Propagation to get the output from the models
                    output = self(features)
                    nan_check(tensor=output, tensor_description='model output tensor')
                    
                    # Computing the loss
                    loss = self.loss_fn(output=output, 
                                        target=mask, 
                                        num_classes=self.label_channels, 
                                        weights=self.dice_penalty_dict, 
                                        class_list=self.data.class_list, 
                                        to_scalar=False, 
                                        **self.loss_function_kwargs)
                    nan_check(tensor=loss, tensor_description='model loss tensor')

                    # Back Propagation for model to learn    
                    loss.backward()
                    #Updating the weight values
                    self.optimizer.step()
                    #Pushing the dice to the cpu and only taking its value
                    loss = loss.cpu().data.item()
                    total_loss += loss

                    batch_num += 1

        # Now we need to prepare the tensor keys for returning
        
        # loss metric
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {
            TensorKey('loss', origin, round_num, True, ('metric',)): np.array((total_loss / num_batches))
        }

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        # Create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in global_model_dict.items()
        }
        # Create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num, False, tags):
                nparray for tensor_name, nparray in local_model_dict.items()
        }
        # The train/validate aggregated function of the next round will look
        # for the updated model parameters.
        # This ensures they will be resolved locally
        next_local_tensorkey_model_dict = {
            TensorKey(tensor_name, origin, round_num + 1, False, ('model',)): nparray
            for tensor_name, nparray in local_model_dict.items()}

        global_tensor_dict = {
            **output_metric_dict,
            **global_tensorkey_model_dict
        }
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict
        }

        # This will signal that the optimizer values are now present,
        # and can be loaded when the model is rebuilt
        self.train_round_completed = True

        # FIXME: these should be filtered earlier:
        # remove everything that starts with '__opt_state'
        keys = list(global_tensor_dict.keys())
        for k in keys:
            if k.tensor_name.startswith('__opt_state'):
                global_tensor_dict.pop(k)
        keys = list(local_tensor_dict.keys())
        for k in keys:
            if k.tensor_name.startswith('__opt_state'):
                local_tensor_dict.pop(k)

        # Return global_tensor_dict, local_tensor_dict
        return global_tensor_dict, local_tensor_dict

    def validate(self,
                 col_name,
                 round_num,
                 input_tensor_dict,
                 use_tqdm=False,
                 **kwargs):

        self.rebuild_model(round_num, input_tensor_dict, validation=True)

        # dice results are dictionaries (keys provided by self.validation_output_keys)
        valscores = {key: [] for key in self.validation_output_keys}
        
        # FeTS Challenge val_dict
        val_dict = {}
        total_samples = 0

        val_loader = self.data.get_val_loader()

        if val_loader == []:
            raise RuntimeError("Attempting to run validation with an empty val loader.")

        if use_tqdm:
            val_loader = tqdm.tqdm(val_loader, desc="validate")

        for subject in val_loader:
            # this is when we are using pt_brainmagedata
            if ('features' in subject.keys()) and ('gt' in subject.keys()):
                features = subject['features']
                nan_check(tensor=features, tensor_description='features tensor')
                mask = subject['gt']
                nan_check(tensor=mask, tensor_description='ground truth mask tensor')
        
                self.sanity_check_val_input_shape(features)
                output = self.infer_batch_with_no_numpy_conversion(features=features)
                nan_check(tensor=output, tensor_description='model output tensor')
                self.sanity_check_val_output_shape(output)
                    
            # using the gandlf loader   
            else:
                self.logger.debug("Validating with subject: {}".format(subject['subject_id']))
                features = torch.cat([subject[key][torchio.DATA] for key in self.channel_keys], dim=1).float()
                nan_check(tensor=features, tensor_description='features tensor')
                mask = subject['label'][torchio.DATA]
                nan_check(tensor=mask, tensor_description='ground truth mask tensor')

                if self.validate_without_patches:
                    self.sanity_check_val_input_shape(features)
                    output = self.data.infer_with_crop(model_inference_function=[self.infer_batch_with_no_numpy_conversion], 
                                                       features=features)
                    nan_check(tensor=output, tensor_description='model output tensor')
                    self.sanity_check_val_output_shape(output)
                else:
                    self.sanity_check_val_input_shape(features)
                    output = self.data.infer_with_crop_and_patches(model_inference_function=[self.infer_batch_with_no_numpy_conversion], 
                                                                   features=features)
                    nan_check(tensor=output, tensor_description='model output tensor')
                    self.sanity_check_val_output_shape(output)
                                    
            # one-hot encoding of ground truth
            mask = one_hot(mask, self.data.class_list).float()
            nan_check(tensor=mask, tensor_description='one_hot ground truth mask tensor')
            
            # sanity check that the output and mask have the same shape
            if output.shape != mask.shape:
                raise ValueError('Model output and ground truth mask are not the same shape.')

            # FIXME: Create a more general losses.py module (with composability and aggregation)
            current_valscore = self.validation_function(output=output, 
                                                        target=mask, 
                                                        class_list=self.data.class_list, 
                                                        fine_grained=self.validate_with_fine_grained_dice, 
                                                        **self.validation_function_kwargs)
            for key, value in current_valscore.items():
                nan_check(tensor=torch.Tensor([value]), tensor_description='validation result with key {}'.format(key))

            # the dice results here are dictionaries (sum up the totals)
            for key in self.validation_output_keys:
                valscores[key].append(current_valscore[key])

            # FeTS Challenge addition:
            # now we call the additional validation functions from the competitor
            for name, func in self.validation_functions:
                score = func(mask.cpu().numpy(), output.cpu().numpy())
                if name not in val_dict:
                    val_dict[name] = score
                else:
                    val_dict[name] += score
            total_samples += 1

        if total_samples != 0:
            for key in val_dict.keys():
                val_dict[key] /= total_samples

        # silly adaptation from brainmage logic to fets challenge logic
        for key, scores_list in valscores.items():
            val_dict['performance_evaluation_metric_' + key] = np.mean(scores_list)

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)

        output_tensor_dict = {}
        for k, v in val_dict.items():
            output_tensor_dict[TensorKey(k, origin, round_num, True, tags)] = np.array(v)

        # Empty list represents metrics that should only be stored locally
        return output_tensor_dict, {}

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        required = super().get_required_tensorkeys_for_function(func_name, **kwargs)
        if func_name == 'train_batches':
            round_number = required[0].round_number
            for hparam in TRAINING_HPARAMS:
                required.append(TensorKey(tensor_name=hparam, origin='GLOBAL', round_number=round_number, report=False, tags=('hparam', 'model')))
        return required

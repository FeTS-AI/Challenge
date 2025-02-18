# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GaNDLFTaskRunner module."""

from copy import deepcopy

import numpy as np
import torch as pt

from openfl.utilities.split import split_tensor_dict_for_holdouts
from openfl.utilities import TensorKey

from openfl.federated.task.runner_gandlf import *

from GANDLF.compute.generic             import create_pytorch_objects
from GANDLF.compute.training_loop       import train_network
from GANDLF.compute.forward_pass        import validate_network

from . import TRAINING_HPARAMS

class FeTSChallengeModel(GaNDLFTaskRunner):
    """FeTSChallenge Model class for Federated Learning."""

    def validate(self, col_name, round_num, input_tensor_dict,
                 use_tqdm=False, **kwargs):
        """Validate.
        Run validation of the model on the local data.
        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)
            kwargs:              Key word arguments passed to GaNDLF main_run
        Returns:
            global_output_dict:   Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.model.eval()
        # self.model.to(self.device)

        epoch_valid_loss, epoch_valid_metric = validate_network(self.model,
                                                                self.data_loader.val_dataloader,
                                                                self.scheduler,
                                                                self.params,
                                                                round_num,
                                                                mode="validation")

        self.logger.info(epoch_valid_loss)
        self.logger.info(epoch_valid_metric)

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)

        output_tensor_dict = {}
        output_tensor_dict[TensorKey('valid_loss', origin, round_num, True, tags)] = np.array(epoch_valid_loss)
        for k, v in epoch_valid_metric.items():
            print(f"Testing ->>>> Metric Key {k} Value {v}")
            if isinstance(v, str):
                v = list(map(float, v.split('_')))

            if np.array(v).size == 1:
                output_tensor_dict[TensorKey(f'valid_{k}', origin, round_num, True, tags)] = np.array(v)
            else:
                for idx,label in enumerate([0,1,2,4]):
                    output_tensor_dict[TensorKey(f'valid_{k}_{label}', origin, round_num, True, tags)] = np.array(v[idx])

        return output_tensor_dict, {}

    def inference(self, col_name, round_num, input_tensor_dict,
                 use_tqdm=False, **kwargs):
        """Inference.
        Run inference of the model on the local data (used for final validation)
        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm (bool):     Use tqdm to print a progress bar (Default=True)
            kwargs:              Key word arguments passed to GaNDLF main_run
        Returns:
            global_output_dict:   Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.model.eval()
        # self.model.to(self.device)

        epoch_valid_loss, epoch_valid_metric = validate_network(self.model,
                                                                self.data_loader.val_dataloader,
                                                                self.scheduler,
                                                                self.params,
                                                                round_num,
                                                                mode="inference")

        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)

        output_tensor_dict = {}
        output_tensor_dict[TensorKey('valid_loss', origin, round_num, True, tags)] = np.array(epoch_valid_loss)
        for k, v in epoch_valid_metric.items():
            print(f"Testing ->>>> Metric Key {k} Value {v}")
            if isinstance(v, str):
                v = list(map(float, v.split('_')))
            if np.array(v).size == 1:
                output_tensor_dict[TensorKey(f'valid_{k}', origin, round_num, True, tags)] = np.array(v)
            else:
                for idx,label in enumerate([0,1,2,4]):
                    output_tensor_dict[TensorKey(f'valid_{k}_{label}', origin, round_num, True, tags)] = np.array(v[idx])

        return output_tensor_dict, {}


    def train(self, col_name, round_num, input_tensor_dict, use_tqdm=False, epochs=1, **kwargs):
        """Train batches.
        Train the model on the requested number of batches.
        Args:
            col_name                : Name of the collaborator
            round_num               : What round is it
            input_tensor_dict       : Required input tensors (for model)
            use_tqdm (bool)         : Use tqdm to print a progress bar (Default=True)
            epochs                  : The number of epochs to train
            crossfold_test          : Whether or not to use cross fold trainval/test
                                    to evaluate the quality of the model under fine tuning
                                    (this uses a separate prameter to pass in the data and 
                                    config used)
            crossfold_test_data_csv : Data csv used to define data used in crossfold test.
                                      This csv does not itself define the folds, just
                                      defines the total data to be used.
            crossfold_val_n         : number of folds to use for the train,val level of the nested crossfold.
            corssfold_test_n        : number of folds to use for the trainval,test level of the nested crossfold.
            kwargs                  : Key word arguments passed to GaNDLF main_run
        Returns:
            global_output_dict      : Tensors to send back to the aggregator
            local_output_dict       : Tensors to maintain in the local TensorDB
        """

        # handle the hparams
        epochs_per_round = int(input_tensor_dict.pop('epochs_per_round'))
        learning_rate = float(input_tensor_dict.pop('learning_rate'))

        self.rebuild_model(round_num, input_tensor_dict)
        # set to "training" mode
        self.model.train()

        # Set the learning rate
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate

        for epoch in range(epochs_per_round):
            self.logger.info(f'Run {epoch} epoch of {round_num} round')
            # FIXME: do we want to capture these in an array rather than simply taking the last value?
            epoch_train_loss, epoch_train_metric = train_network(self.model,
                                                                 self.data_loader.train_dataloader,
                                                                 self.optimizer,
                                                                 self.params)

        # output model tensors (Doesn't include TensorKey)
        tensor_dict = self.get_tensor_dict(with_opt_vars=True)

        metric_dict = {'loss': epoch_train_loss}
        for k, v in epoch_train_metric.items():
            print(f"Testing ->>>> Metric Key {k} Value {v}")
            if isinstance(v, str):
                v = list(map(float, v.split('_')))
            if np.array(v).size == 1:
                metric_dict[f'train_{k}'] = np.array(v)
            else:
                for idx,label in enumerate([0,1,2,4]):
                    metric_dict[f'train_{k}_{label}'] = np.array(v[idx])


        # Return global_tensor_dict, local_tensor_dict
        # is this even pt-specific really?
        global_tensor_dict, local_tensor_dict = create_tensorkey_dicts(tensor_dict,
                                                                       metric_dict,
                                                                       col_name,
                                                                       round_num,
                                                                       self.logger,
                                                                       self.tensor_dict_split_fn_kwargs)

        # Update the required tensors if they need to be pulled from the
        # aggregator
        # TODO this logic can break if different collaborators have different
        # roles between rounds.
        # For example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator because
        # these are only created after training occurs. A work around could
        # involve doing a single epoch of training on random data to get the
        # optimizer names, and then throwing away the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        # This will signal that the optimizer values are now present,
        # and can be loaded when the model is rebuilt
        self.train_round_completed = True

        # Return global_tensor_dict, local_tensor_dict
        return global_tensor_dict, local_tensor_dict

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        required = super().get_required_tensorkeys_for_function(func_name, **kwargs)
        if func_name == 'train':
            round_number = required[0].round_number
            for hparam in TRAINING_HPARAMS:
                required.append(TensorKey(tensor_name=hparam, origin='GLOBAL', round_number=round_number, report=False, tags=('hparam', 'model')))
        return required

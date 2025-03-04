import os
from copy import deepcopy
from typing import Union

import numpy as np
import torch as pt
import yaml

from sys import path
from openfl.federated import Plan
from pathlib import Path

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from openfl.databases import TensorDB
from openfl.utilities import TaskResultKey, TensorKey, change_tags

from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.config_manager import ConfigManager

#from .fets_challenge_model import inference, fedavg

def get_metric(metric_name, col_name, fl_round, output_tensor_dict):
    print(f'Getting metric {metric_name} for collaborator {col_name} at round {fl_round}')
    target_tags = ('metric', 'validate')
    tensor_key = TensorKey(metric_name, col_name, fl_round, True, target_tags)

    # Check if the key exists in the dictionary
    value = None
    if tensor_key in output_tensor_dict:
        # Retrieve the value associated with the TensorKey
        value = output_tensor_dict[tensor_key]
        print(value)
    else:
        print(f"TensorKey {tensor_key} not found in the dictionary")
    
    return value

class FeTSFederatedFlow(FLSpec):
    def __init__(self, fets_model, rounds=3, **kwargs):
        super().__init__(**kwargs)
        self.fets_model = fets_model
        self.n_rounds = rounds
        self.current_round = 1

    @aggregator
    def start(self):
        self.collaborators = self.runtime.collaborators
        self.next(self.initialize_collaborators, foreach='collaborators')

    @collaborator
    def initialize_collaborators(self):
        if isinstance(self.gandlf_config, str) and os.path.exists(self.gandlf_config):
            gandlf_conf = yaml.safe_load(open(self.gandlf_config, "r"))

        print(gandlf_conf)

        #gandlf_config_path = "/home/ad_tbanda/code/fedAI/Challenge/Task_1/gandlf_config.yaml"
        gandlf_config = Plan.load(Path(self.gandlf_config))
        print(gandlf_config)
        print(gandlf_config['weighted_loss'])

        gandlf_conf = ConfigManager(self.gandlf_config)

        (
            model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
            params,
        ) = create_pytorch_objects(
            gandlf_conf, train_csv=self.train_csv, val_csv=self.val_csv, device=self.device
        )
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params
        self.device = self.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = 1
        self.tensor_db = TensorDB()
        self.next(self.aggregated_model_validation)

    @collaborator
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input}')
        print(f'Val Loader: {self.val_loader}')
        self.agg_output_dict, _ = self.fets_model.validate(self.model, self.input, self.current_round, self.val_loader, self.params, self.scheduler)
        print(f'{self.input} value of {self.agg_output_dict}')
        self.next(self.train)

    @collaborator
    def train(self):
        print(f'Performing training for collaborator {self.input}')
        global_output_tensor_dict, local_output_tensor_dict =  self.fets_model.train(self.model, self.input, self.current_round, self.train_loader, self.params, self.optimizer, self.epochs)
        self.tensor_db.cache_tensor(global_output_tensor_dict)
        self.tensor_db.cache_tensor(local_output_tensor_dict)
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_output_dict, _ = self.fets_model.validate(self.model, self.input, self.current_round, self.val_loader, self.params, self.scheduler)
        print(f'Doing local model validation for collaborator {self.input}:'
              + f' {self.local_output_dict}')
        self.next(self.join)

    @aggregator
    def join(self, inputs):

        total_loss = 0.0
        total_dice = 0.0
        num_inputs = len(inputs)

        for idx, col in enumerate(inputs):
            print(f'Aggregating results for {idx}')
            round_loss = get_metric('valid_loss', str(idx + 1), self.current_round, col.agg_output_dict)
            round_dice = get_metric('valid_dice', str(idx + 1), self.current_round, col.agg_output_dict)
            dice_label_0 = get_metric('valid_dice_per_label_0', str(idx + 1), self.current_round, col.agg_output_dict)
            dice_label_1 = get_metric('valid_dice_per_label_1', str(idx + 1), self.current_round, col.agg_output_dict)

            print(f'Round loss: {round_loss}')
            print(f'Round dice: {round_dice}')
            print(f'Dice label 0: {dice_label_0}')
            print(f'Dice label 1: {dice_label_1}')

            total_loss += round_loss
            total_dice += round_dice
        # dice_label_0 = get_metric('valid_dice_per_label_0', self.current_round, aggregator.tensor_db)
        # dice_label_1 = get_metric('valid_dice_per_label_1', self.current_round, aggregator.tensor_db)
        # dice_label_2 = get_metric('valid_dice_per_label_2', self.current_round, aggregator.tensor_db)
        # dice_label_4 = get_metric('valid_dice_per_label_4', self.current_round, aggregator.tensor_db)
        #self.model = fedavg([input.model for input in inputs])

        average_round_loss = total_loss / num_inputs
        average_round_dice = total_dice / num_inputs

        print(f'Average round loss: {average_round_loss}')
        print(f'Average round dice: {average_round_dice}')

        # times_per_collaborator = compute_times_per_collaborator(collaborator_names,
        #                                                         training_collaborators,
        #                                                         epochs_per_round,
        #                                                         collaborator_data_loaders,
        #                                                         collaborator_time_stats,
        #                                                         round_num)
        # collaborator_times_per_round[round_num] = times_per_collaborator

        total_simulated_time = 0
        best_dice = -1.0
        best_dice_over_time_auc = 0

        # times_list = [(t, col) for col, t in times_per_collaborator.items()]
        # times_list = sorted(times_list)

        # the round time is the max of the times_list
        # round_time = max([t for t, _ in times_list])
        # total_simulated_time += round_time

        if best_dice < average_round_dice:
            best_dice = average_round_dice
            # Set the weights for the final model
            # if round_num == 0:
            #     # here the initial model was validated (temp model does not exist)
            #     logger.info(f'Skipping best model saving to disk as it is a random initialization.')
            # elif not os.path.exists(f'checkpoint/{checkpoint_folder}/temp_model.pkl'):
            #     raise ValueError(f'Expected temporary model at: checkpoint/{checkpoint_folder}/temp_model.pkl to exist but it was not found.')
            # else:
            #     # here the temp model was the one validated
            #     shutil.copyfile(src=f'checkpoint/{checkpoint_folder}/temp_model.pkl',dst=f'checkpoint/{checkpoint_folder}/best_model.pkl')
            #     logger.info(f'Saved model with best average binary DICE: {best_dice} to ~/.local/workspace/checkpoint/{checkpoint_folder}/best_model.pkl')

        ## CONVERGENCE METRIC COMPUTATION
        # update the auc score
        # best_dice_over_time_auc += best_dice * round_time

        # project the auc score as remaining time * best dice
        # this projection assumes that the current best score is carried forward for the entire week
        # projected_auc = (MAX_SIMULATION_TIME - total_simulated_time) * best_dice + best_dice_over_time_auc
        # projected_auc /= MAX_SIMULATION_TIME

        # # End of round summary
        # summary = '"**** END OF ROUND {} SUMMARY *****"'.format(self.current_round)
        # summary += "\n\tSimulation Time: {} minutes".format(round(total_simulated_time / 60, 2))
        # summary += "\n\t(Projected) Convergence Score: {}".format(projected_auc)
        # summary += "\n\tDICE Label 0: {}".format(dice_label_0)
        # summary += "\n\tDICE Label 1: {}".format(dice_label_1)
        # summary += "\n\tDICE Label 2: {}".format(dice_label_2)
        # summary += "\n\tDICE Label 4: {}".format(dice_label_4)
        # if include_validation_with_hausdorff:
        #     summary += "\n\tHausdorff95 Label 0: {}".format(hausdorff95_label_0)
        #     summary += "\n\tHausdorff95 Label 1: {}".format(hausdorff95_label_1)
        #     summary += "\n\tHausdorff95 Label 2: {}".format(hausdorff95_label_2)
        #     summary += "\n\tHausdorff95 Label 4: {}".format(hausdorff95_label_4)

        # [TODO] : Aggregation Function

        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        if self.current_round == self.n_rounds:
            self.next(self.end)
        else:
            self.current_round += 1
            self.next(self.aggregated_model_validation, foreach='collaborators')

    @aggregator
    def end(self):
        print('This is the end of the flow')
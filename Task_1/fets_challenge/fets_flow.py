import os
from copy import deepcopy
from typing import Union

import logging
import pandas as pd
import numpy as np
import torch as pt
import yaml
import shutil

from sys import path
from openfl.federated import Plan
from pathlib import Path

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from openfl.databases import TensorDB
from openfl.utilities import TaskResultKey, TensorKey, change_tags
from .checkpoint_utils import setup_checkpoint_folder, save_checkpoint, load_checkpoint

from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_metric(metric_name, fl_round, agg_tensor_db):
    target_tags = ('metric', 'validate_agg')
    metric_tensor_key = TensorKey(metric_name, 'aggregator', fl_round, True, target_tags)
    logger.info(f'Getting metric {metric_name} at round {fl_round} tensor key: {metric_tensor_key}')
    nparray = agg_tensor_db.get_tensor_from_cache(metric_tensor_key)
    #logger.info(f'nparray for {metric_name} at round {fl_round}: {nparray.item()}')
    return nparray.item()

def cache_tensor_dict(tensor_dict, agg_tensor_db, idx, agg_out_dict):
    for key, value in tensor_dict.items():
        new_tags = change_tags(key.tags, add_field=str(idx + 1))
        modified_key = TensorKey(
            tensor_name=key.tensor_name,
            origin="aggregator",
            round_number=key.round_number,
            report=key.report,
            tags=new_tags
        )
        agg_out_dict[modified_key] = value
    agg_tensor_db.cache_tensor(agg_out_dict)

class FeTSFederatedFlow(FLSpec):
    def __init__(self, fets_model, rounds=3, **kwargs):
        super().__init__(**kwargs)
        self.fets_model = fets_model
        self.n_rounds = rounds
        self.current_round = 1

    @aggregator
    def start(self):
        self.collaborators = self.runtime.collaborators
        logger.info(f'Collaborators: {self.collaborators}')
        logger.info(f'save_checkpoints: {self.save_checkpoints}')
        logger.info(f'collaborator_time_stats: {self.collaborator_time_stats}')
        logger.info(f'restore_from_checkpoint_folder: {self.restore_from_checkpoint_folder}')

        self.experiment_results = {
            'round':[],
            'time': [],
            'convergence_score': [],
            'round_dice': [],
            'dice_label_0': [],
            'dice_label_1': [],
            'dice_label_2': [],
            'dice_label_4': [],
        }
        if self.include_validation_with_hausdorff:
            self.experiment_results.update({
                'hausdorff95_label_0': [],
                'hausdorff95_label_1': [],
                'hausdorff95_label_2': [],
                'hausdorff95_label_4': [],
            })

        self.total_simulated_time = 0
        self.best_dice = -1.0
        self.best_dice_over_time_auc = 0

        self.checkpoint_folder = ""
        self.collaborators_chosen_each_round = {}
        self.collaborator_times_per_round = {}
        if self.restore_from_checkpoint_folder is None:
            self.checkpoint_folder = setup_checkpoint_folder()
            logger.info(f'\nCreated experiment folder {self.checkpoint_folder}...')
            starting_round_num = 0
        else:
            if not Path(f'checkpoint/{self.restore_from_checkpoint_folder}').exists():
                logger.warning(f'Could not find provided checkpoint folder: {self.restore_from_checkpoint_folder}. Exiting...')
                exit(1)
            else:
                logger.info(f'Attempting to load last completed round from {self.restore_from_checkpoint_folder}')
                state = load_checkpoint(self.restore_from_checkpoint_folder)
                self.checkpoint_folder = self.restore_from_checkpoint_folder

                [loaded_collaborator_names, starting_round_num, self.collaborator_time_stats,
                self.total_simulated_time, self.best_dice, self.best_dice_over_time_auc,
                self.collaborators_chosen_each_round, self.collaborator_times_per_round,
                self.experiment_results, summary, agg_tensor_db] = state

                if loaded_collaborator_names != self.collaborator_names:
                    logger.error(f'Collaborator names found in checkpoint ({loaded_collaborator_names}) '
                                f'do not match provided collaborators ({self.collaborator_names})')
                    exit(1)

                logger.info(f'Previous summary for round {starting_round_num}')
                logger.info(summary)

                starting_round_num += 1
                #self.tensor_db.tensor_db = agg_tensor_db
                self.round_number = starting_round_num
        self.next(self.fetch_hyper_parameters)
    
    @aggregator
    def fetch_hyper_parameters(self):
        print("*" * 40)
        print("Starting round  {}".format(self.current_round))
        print("*" * 40)
        logger.info('Fetching hyperparameters')
        tensrdb = TensorDB()
        hparams = self.training_hyper_parameters_for_round(self.collaborators,
                                                            tensrdb._iterate(),
                                                            self.current_round,
                                                            self.collaborators_chosen_each_round,
                                                            self.collaborator_times_per_round)

        learning_rate, epochs_per_round = hparams

        if (epochs_per_round is None):
            logger.warning('Hyper-parameter function warning: function returned None for "epochs_per_round". Setting "epochs_per_round" to 1')
            epochs_per_round = 1
        
        hparam_message = "\n\tlearning rate: {}".format(learning_rate)

        hparam_message += "\n\tepochs_per_round: {}".format(epochs_per_round)

        logger.info("Hyper-parameters for round {}:{}".format(self.current_round, hparam_message))

        # cache each tensor in the aggregator tensor_db
        self.hparam_dict = {}
        tk = TensorKey(tensor_name='learning_rate',
                        origin=self.uuid,
                        round_number=self.current_round,
                        report=False,
                        tags=('hparam', 'model'))
        self.hparam_dict[tk] = np.array(learning_rate)
        tk = TensorKey(tensor_name='epochs_per_round',
                        origin=self.uuid,
                        round_number=self.current_round,
                        report=False,
                        tags=('hparam', 'model'))
        self.hparam_dict[tk] = np.array(epochs_per_round)



        # times_per_collaborator = compute_times_per_collaborator(collaborator_names,
        #                                                         training_collaborators,
        #                                                         epochs_per_round,
        #                                                         collaborator_data_loaders,
        #                                                         collaborator_time_stats,
        #                                                         round_num)


        if self.current_round == 1:
            logger.info('[Next Step] : Initializing collaborators')
            self.next(self.initialize_colls, foreach='collaborators')
        else:
            logger.info('[Next Step] : Aggregated model validation')
            self.next(self.aggregated_model_validation, foreach='collaborators')
        

    @collaborator
    def initialize_colls(self):
        logger.info(f'Initializing collaborator {self.input}')
        if isinstance(self.gandlf_config, str) and os.path.exists(self.gandlf_config):
            gandlf_conf = yaml.safe_load(open(self.gandlf_config, "r"))

        logger.info(gandlf_conf)
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
        self.next(self.aggregated_model_validation)

    # @collaborator
    # def init_tensors(self):
    #     logger.info(f'Initializing tensors for collaborator {self.input}')
    #     coll_tensor_dict = self.fets_model.get_tensor_dict(self.model)
    #     # for key, value in coll_tensor_dict.items():
    #     #     print(f'Adding tensor {key}')
    #     #     print(f'Value of tensor {key} is {value}')

    #     self.fets_model.rebuild_model(self.model, self.current_round, coll_tensor_dict, "cpu")
    #     self.next(self.aggregated_model_validation)

    @collaborator
    def aggregated_model_validation(self):
        logger.info(f'Performing aggregated model validation for collaborator {self.input}')
        self.agg_valid_dict, _ = self.fets_model.validate(self.model, self.input, self.current_round, self.val_loader, self.params, self.scheduler, apply="global")
        #logger.info(f'{self.input} value of {self.agg_valid_dict.keys()}')
        self.next(self.train)

    @collaborator
    def train(self):
        logger.info(f'Performing training for collaborator {self.input}')
        self.global_output_tensor_dict, local_output_tensor_dict =  self.fets_model.train(self.model, self.input, self.current_round, self.train_loader, self.params, self.optimizer, self.hparam_dict, self.epochs)
        #logger.info(f'{self.input} value of {self.global_output_tensor_dict.keys()}')
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        logger.info(f'Performing local model validation for collaborator {self.input}')
        self.local_valid_dict, _ = self.fets_model.validate(self.model, self.input, self.current_round, self.val_loader, self.params, self.scheduler, apply="local")
        #logger.info(f'Doing local model validation for collaborator {self.input}:' + f' {self.local_output_dict}')
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        agg_tensor_db = TensorDB()
        tensor_keys_per_col = {}
        for idx, col in enumerate(inputs):
            logger.info(f'Aggregating results for {idx}')
            agg_out_dict = {}
            cache_tensor_dict(col.local_valid_dict, agg_tensor_db, idx, agg_out_dict)
            cache_tensor_dict(col.agg_valid_dict, agg_tensor_db, idx, agg_out_dict)
            cache_tensor_dict(col.global_output_tensor_dict, agg_tensor_db, idx, agg_out_dict)

            # Store the keys for each collaborator
            tensor_keys = []
            for tensor_key in agg_out_dict.keys():
                #logger.info(f'Adding tensor key {tensor_key} to the dict of tensor keys')
                tensor_keys.append(tensor_key)
                tensor_keys_per_col[str(idx + 1)] = tensor_keys

        # [TODO] : Aggregation Function -> Collaborator Weight Dict
        collaborator_weight_dict = {'1': 0.3333333333333333, '2': 0.3333333333333333, '3': 0.3333333333333333}
        for col,tensor_keys in tensor_keys_per_col.items():
            for tensor_key in tensor_keys:
                tensor_name, origin, round_number, report, tags = tensor_key
                #logger.info(f'Aggregating tensor {tensor_name} from collaborator {origin} for round {round_number}')
                new_tags = change_tags(tags, remove_field=col)
                agg_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
                # Aggregates the tensor values for the tensor key and stores it in tensor_db
                agg_results = agg_tensor_db.get_aggregated_tensor(
                    agg_tensor_key,
                    collaborator_weight_dict,
                    aggregation_function=self.aggregation_type,
                )
                #logger.info(f'Aggregated tensor value for tensor key {agg_tensor_key}')

        agg_tensor_dict = {}
        for col,tensor_keys in tensor_keys_per_col.items():
            for tensor_key in tensor_keys:
                tensor_name, origin, round_number, report, tags = tensor_key
                #logger.info(f'Training tensor_key {tensor_key}')
                if 'trained' in tags:
                    #logger.info(f'Fetching tensor {tensor_name} from tensor_db for round {round_number}')
                    new_tags = change_tags(tags, remove_field=col)
                    new_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
                    if tensor_name not in agg_tensor_dict:
                        agg_tensor_dict[tensor_name] = agg_tensor_db.get_tensor_from_cache(new_tensor_key)
                        #logger.info(f'Fetched tensor {tensor_name} from tensor_db for round {round_number}')

        # Rebuild the model with the aggregated tensor_dict
        for input in inputs:
            logger.info(f'Updating model for collaborator {input}')
            local_tensor_dict = deepcopy(agg_tensor_dict)
            self.fets_model.rebuild_model(input.model, self.current_round, local_tensor_dict, "cpu")
            local_tensor_dict = None

        round_loss = get_metric('valid_loss', self.current_round, agg_tensor_db)
        round_dice = get_metric('valid_dice', self.current_round, agg_tensor_db)
        dice_label_0 = get_metric('valid_dice_per_label_0', self.current_round, agg_tensor_db)
        dice_label_1 = get_metric('valid_dice_per_label_1', self.current_round, agg_tensor_db)
        dice_label_2 = get_metric('valid_dice_per_label_2', self.current_round, agg_tensor_db)
        dice_label_4 = get_metric('valid_dice_per_label_4', self.current_round, agg_tensor_db)
        if self.include_validation_with_hausdorff:
            hausdorff95_label_0 = get_metric('valid_hd95_per_label_0', self.current_round, agg_tensor_db)
            hausdorff95_label_1 = get_metric('valid_hd95_per_label_1', self.current_round, agg_tensor_db)
            hausdorff95_label_2 = get_metric('valid_hd95_per_label_2', self.current_round, agg_tensor_db)
            hausdorff95_label_4 = get_metric('valid_hd95_per_label_4', self.current_round, agg_tensor_db)

        # times_list = [(t, col) for col, t in times_per_collaborator.items()]
        # times_list = sorted(times_list)

        # the round time is the max of the times_list
        # round_time = max([t for t, _ in times_list])
        # self.total_simulated_time += round_time

        if self.best_dice < round_dice:
            self.best_dice = round_dice
            # Set the weights for the final model
            if self.current_round == 0:
                # here the initial model was validated (temp model does not exist)
                logger.info(f'Skipping best model saving to disk as it is a random initialization.')
            elif not os.path.exists(f'checkpoint/{self.checkpoint_folder}/temp_model.pkl'):
                raise ValueError(f'Expected temporary model at: checkpoint/{self.checkpoint_folder}/temp_model.pkl to exist but it was not found.')
            else:
                # here the temp model was the one validated
                shutil.copyfile(src=f'checkpoint/{self.checkpoint_folder}/temp_model.pkl',dst=f'checkpoint/{self.checkpoint_folder}/best_model.pkl')
                logger.info(f'Saved model with best average binary DICE: {self.best_dice} to ~/.local/workspace/checkpoint/{self.checkpoint_folder}/best_model.pkl')

        ## CONVERGENCE METRIC COMPUTATION
        # update the auc score
        self.best_dice_over_time_auc += self.best_dice * self.current_round

        # project the auc score as remaining time * best dice
        # this projection assumes that the current best score is carried forward for the entire week
        projected_auc = (self.max_simulation_time - self.total_simulated_time) * self.best_dice + self.best_dice_over_time_auc
        projected_auc /= self.max_simulation_time

        # # End of round summary
        summary = '"**** END OF ROUND {} SUMMARY *****"'.format(self.current_round)
        summary += "\n\tSimulation Time: {} minutes".format(round(self.total_simulated_time / 60, 2))
        summary += "\n\t(Projected) Convergence Score: {}".format(projected_auc)
        summary += "\n\tRound Loss: {}".format(round_loss)
        summary += "\n\Round Dice: {}".format(round_dice)
        summary += "\n\tDICE Label 0: {}".format(dice_label_0)
        summary += "\n\tDICE Label 1: {}".format(dice_label_1)
        summary += "\n\tDICE Label 2: {}".format(dice_label_2)
        summary += "\n\tDICE Label 4: {}".format(dice_label_4)
        if self.include_validation_with_hausdorff:
            summary += "\n\tHausdorff95 Label 0: {}".format(hausdorff95_label_0)
            summary += "\n\tHausdorff95 Label 1: {}".format(hausdorff95_label_1)
            summary += "\n\tHausdorff95 Label 2: {}".format(hausdorff95_label_2)
            summary += "\n\tHausdorff95 Label 4: {}".format(hausdorff95_label_4)
        logger.info(summary)

        self.experiment_results['round'].append(self.current_round)
        self.experiment_results['time'].append(self.total_simulated_time)
        self.experiment_results['convergence_score'].append(projected_auc)
        self.experiment_results['round_dice'].append(round_dice)
        self.experiment_results['dice_label_0'].append(dice_label_0)
        self.experiment_results['dice_label_1'].append(dice_label_1)
        self.experiment_results['dice_label_2'].append(dice_label_2)
        self.experiment_results['dice_label_4'].append(dice_label_4)
        if self.include_validation_with_hausdorff:
            self.experiment_results['hausdorff95_label_0'].append(hausdorff95_label_0)
            self.experiment_results['hausdorff95_label_1'].append(hausdorff95_label_1)
            self.experiment_results['hausdorff95_label_2'].append(hausdorff95_label_2)
            self.experiment_results['hausdorff95_label_4'].append(hausdorff95_label_4)

        if self.save_checkpoints:
            logger.info(f'Saving checkpoint for round {self.current_round} : checkpoint folder {self.checkpoint_folder}')
            logger.info(f'To resume from this checkpoint, set the restore_from_checkpoint_folder parameter to \'{self.checkpoint_folder}\'')
            save_checkpoint(self.checkpoint_folder, agg_tensor_db,
                            self.collaborator_names, self.runtime.collaborators,
                            self.current_round, self.collaborator_time_stats,
                            self.total_simulated_time, self.best_dice,
                            self.best_dice_over_time_auc,
                            self.collaborators_chosen_each_round,
                            self.collaborator_times_per_round,
                            self.experiment_results,
                            summary)

        # if the total_simulated_time has exceeded the maximum time, we break
        # in practice, this means that the previous round's model is the last model scored,
        # so a long final round should not actually benefit the competitor, since that final
        # model is never globally validated
        if self.total_simulated_time > self.max_simulation_time:
            logger.info("Simulation time exceeded. Ending Experiment")
            self.next(self.end)

        # save the most recent aggregated model in native format to be copied over as best when appropriate
        # (note this model has not been validated by the collaborators yet)
        # self.fets_model.rebuild_model(round_num, aggregator.last_tensor_dict, validation=True)
        self.fets_model.save_native(f'checkpoint/{self.checkpoint_folder}/temp_model.pkl')

        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        if self.current_round == self.n_rounds:
            print('************* EXPERIMENT COMPLETED *************')
            print('Experiment results:')
            print(pd.DataFrame.from_dict(self.experiment_results))
            self.next(self.end)
        else:
            self.current_round += 1
            self.next(self.fetch_hyper_parameters)

    @aggregator
    def end(self):
        logger.info('This is the end of the flow')

"""FeTS Federated Flow."""

import os
import shutil
import time
import logging
from copy import deepcopy
import pandas as pd
from pathlib import Path
import torch

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from openfl.databases import TensorDB
from openfl.utilities import TensorKey, change_tags

from .fets_data_loader import FeTSDataLoader

from .checkpoint_utils import setup_checkpoint_folder, save_checkpoint, load_checkpoint
from .time_utils import gen_collaborator_time_stats, compute_times_per_collaborator, MAX_SIMULATION_TIME

from GANDLF.compute.generic import create_pytorch_objects

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# [TODO] - FixMe Dataloaders cannot be passed as private attributes of collaborator.
# This is a temporary workaround to store the dataloaders in a global variable.
collaborator_data_loaders = {}

class FeTSFederatedFlow(FLSpec):
    def __init__(self, fets_model, params_dict, rounds=5 , device="cpu",  **kwargs):
        super().__init__(**kwargs)
        self.fets_model = fets_model
        self.n_rounds = rounds
        self.device = device
        self.current_round = 0
        self.total_simulated_time = 0
        self.best_dice = -1.0
        self.best_dice_over_time_auc = 0
        self.collaborators_chosen_each_round = {}
        self.collaborator_times_per_round = {}
        self.agg_tensor_dict = {}
        self.restored = False

        self.include_validation_with_hausdorff = params_dict.get('include_validation_with_hausdorff', False)
        self.use_pretrained_model = params_dict.get('use_pretrained_model', False)
        self.choose_training_collaborators = params_dict.get('choose_training_collaborators', None)
        self.training_hyper_parameters_for_round = params_dict.get('training_hyper_parameters_for_round', None)
        self.restore_from_checkpoint_folder = params_dict.get('restore_from_checkpoint_folder', None)
        self.save_checkpoints = params_dict.get('save_checkpoints', False)

        # GaNDLF config
        self.gandlf_config = params_dict.get('gandlf_config', None)

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

    def _get_metric(self, metric_name, fl_round, agg_tensor_db):
        tensor_key = TensorKey(metric_name, 'aggregator', fl_round, True, ('metric', 'validate_agg'))
        return agg_tensor_db.get_tensor_from_cache(tensor_key).item()

    def _cache_tensor_dict(self, tensor_dict, agg_tensor_db, idx, agg_out_dict):
        agg_out_dict.update({
            TensorKey(
                tensor_name=key.tensor_name,
                origin="aggregator",
                round_number=key.round_number,
                report=key.report,
                tags=change_tags(key.tags, add_field=str(idx + 1))
            ): value
            for key, value in tensor_dict.items()
        })
        # Cache the updated dictionary in agg_tensor_db
        agg_tensor_db.cache_tensor(agg_out_dict)

    def _get_aggregated_dict_with_tensorname(self, agg_tensor_dict, current_round=0, lookup_tags='aggregated'):
        return {
            tensor_key.tensor_name: value
            for tensor_key, value in agg_tensor_dict.items()
            if lookup_tags in tensor_key.tags
        }

    def _update_metrics(self, current_round, agg_tensor_db, experiment_results, include_validation_with_hausdorff, 
                    total_simulated_time, projected_auc):

        dice_metrics = [
            'valid_loss', 'valid_dice', 
            'valid_dice_per_label_0', 'valid_dice_per_label_1', 
            'valid_dice_per_label_2', 'valid_dice_per_label_4'
        ]
        hausdorff_metrics = [
            'valid_hd95_per_label_0', 'valid_hd95_per_label_1', 
            'valid_hd95_per_label_2', 'valid_hd95_per_label_4'
        ]

        # Fetch dice metrics
        dice_values = {metric: self._get_metric(metric, current_round, agg_tensor_db) for metric in dice_metrics}

        # Fetch Hausdorff metrics if required
        hausdorff_values = {}
        if include_validation_with_hausdorff:
            hausdorff_values = {metric: self._get_metric(metric, current_round, agg_tensor_db) for metric in hausdorff_metrics}
        
        # # End of round summary        
        summary = '"**** END OF ROUND {} SUMMARY *****"'.format(current_round)
        summary += "\n\tSimulation Time: {} minutes".format(round(total_simulated_time / 60, 2))
        summary += "\n\t(Projected) Convergence Score: {}".format(projected_auc)
        summary += "\n\tRound Loss: {}".format(dice_values['valid_loss'])
        summary += "\n\tRound Dice: {}".format(dice_values['valid_dice'])
        summary += "\n\tDICE Label 0: {}".format(dice_values['valid_dice_per_label_0'])
        summary += "\n\tDICE Label 1: {}".format(dice_values['valid_dice_per_label_1'])
        summary += "\n\tDICE Label 2: {}".format(dice_values['valid_dice_per_label_2'])
        summary += "\n\tDICE Label 4: {}".format(dice_values['valid_dice_per_label_4'])
        if include_validation_with_hausdorff:
            summary += "\n\tHausdorff95 Label 0: {}".format(hausdorff_values['valid_hd95_per_label_0'])
            summary += "\n\tHausdorff95 Label 1: {}".format(hausdorff_values['valid_hd95_per_label_1'])
            summary += "\n\tHausdorff95 Label 2: {}".format(hausdorff_values['valid_hd95_per_label_2'])
            summary += "\n\tHausdorff95 Label 4: {}".format(hausdorff_values['valid_hd95_per_label_4'])
        logger.info(summary)

        experiment_results['round'].append(current_round)
        experiment_results['time'].append(total_simulated_time)
        experiment_results['convergence_score'].append(projected_auc)
        experiment_results['round_dice'].append(dice_values['valid_dice'])
        experiment_results['dice_label_0'].append(dice_values['valid_dice_per_label_0'])
        experiment_results['dice_label_1'].append(dice_values['valid_dice_per_label_1'])
        experiment_results['dice_label_2'].append(dice_values['valid_dice_per_label_2'])
        experiment_results['dice_label_4'].append(dice_values['valid_dice_per_label_4'])
        if include_validation_with_hausdorff:
            experiment_results['hausdorff95_label_0'].append(hausdorff_values['valid_hd95_per_label_0'])
            experiment_results['hausdorff95_label_1'].append(hausdorff_values['valid_hd95_per_label_1'])
            experiment_results['hausdorff95_label_2'].append(hausdorff_values['valid_hd95_per_label_2'])
            experiment_results['hausdorff95_label_4'].append(hausdorff_values['valid_hd95_per_label_4'])

        return summary, dice_values['valid_dice']

    def _initialize_aggregator_model(self):
        """Initialize the aggregator model and its components."""
        model, optimizer, _, _, scheduler, params = create_pytorch_objects(
            self.gandlf_config, None, None, device=self.device
        )
        self.fets_model.model = model
        self.fets_model.optimizer = optimizer
        self.fets_model.scheduler = scheduler
        self.fets_model.params = params

    def _restore_from_checkpoint(self):
        """Restore the experiment state from a checkpoint."""
        checkpoint_path = Path(f'checkpoint/{self.restore_from_checkpoint_folder}')
        if not checkpoint_path.exists():
            logger.warning(f'Could not find provided checkpoint folder: {self.restore_from_checkpoint_folder}. Exiting...')
            exit(1)

        logger.info(f'Attempting to load last completed round from {self.restore_from_checkpoint_folder}')
        state = load_checkpoint(self.restore_from_checkpoint_folder)
        self.checkpoint_folder = self.restore_from_checkpoint_folder

        (
            loaded_collaborator_names, starting_round_num, self.collaborator_time_stats,
            self.total_simulated_time, self.best_dice, self.best_dice_over_time_auc,
            self.collaborators_chosen_each_round, self.collaborator_times_per_round,
            self.experiment_results, summary, agg_tensor_db
        ) = state

        if loaded_collaborator_names != self.collaborator_names:
            logger.error(f'Collaborator names found in checkpoint ({loaded_collaborator_names}) '
                        f'do not match provided collaborators ({self.collaborator_names})')
            exit(1)

        self.restored = True
        logger.info(f'Previous summary for round {starting_round_num}')
        logger.info(summary)

        # Update the agg_tensor_dict from stored tensor_db
        self.current_round = starting_round_num
        self._load_agg_tensor_dict(agg_tensor_db)

    def _setup_new_experiment(self):
        """Set up a new experiment folder and initialize the tensor dictionary."""
        self.checkpoint_folder = setup_checkpoint_folder()
        logger.info(f'\nCreated experiment folder {self.checkpoint_folder}...')
        self.current_round = 0

        # Initialize the tensor dictionary for the first round
        tensor_dict = self.fets_model.get_tensor_dict()
        self.agg_tensor_dict.update({
            TensorKey(
                tensor_name=key,
                origin='aggregator',
                round_number=self.current_round,
                report=False,
                tags=('aggregated',)
            ): value
            for key, value in tensor_dict.items()
        })
    
    def _load_agg_tensor_dict(self, agg_tensor_db):
        """Load the agg_tensor_dict from the stored tensor_db."""
        for _, record in agg_tensor_db.iterrows():
            tensor_key = TensorKey(
                record["tensor_name"], record["origin"], record["round"],
                record["report"], record["tags"]
            )
            self.agg_tensor_dict[tensor_key] = record["nparray"]

    def _aggregate_tensors(self, agg_tensor_db, tensor_keys_per_col, collaborator_weight_dict):
        """Aggregate tensors and cache the results."""
        self.aggregation_type.set_state_data_for_round(self.collaborators_chosen_each_round, self.collaborator_times_per_round)
        for col, tensor_keys in tensor_keys_per_col.items():
            for tensor_key in tensor_keys:
                tensor_name, origin, round_number, report, tags = tensor_key
                if col in tags:
                    new_tags = change_tags(tags, remove_field=col)
                    agg_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
                    if agg_tensor_db.get_tensor_from_cache(agg_tensor_key) is None:
                        agg_results = agg_tensor_db.get_aggregated_tensor(
                            agg_tensor_key,
                            collaborator_weight_dict,
                            aggregation_function=self.aggregation_type,
                        )
                        agg_tag_tk = TensorKey(tensor_name, origin, round_number, report, ('aggregated',))
                        agg_tensor_db.cache_tensor({agg_tag_tk: agg_results})

    def _process_collaborators(self, inputs, agg_tensor_db, collaborator_weights_unnormalized, times_per_collaborator):
        """Process tensors for each collaborator and cache them."""
        tensor_keys_per_col = {}
        for idx, col in enumerate(inputs):
            agg_out_dict = {}
            self._cache_tensor_dict(col.local_valid_dict, agg_tensor_db, idx, agg_out_dict)
            self._cache_tensor_dict(col.agg_valid_dict, agg_tensor_db, idx, agg_out_dict)
            self._cache_tensor_dict(col.global_output_tensor_dict, agg_tensor_db, idx, agg_out_dict)

            # Store the keys for each collaborator
            tensor_keys_per_col[str(idx + 1)] = list(agg_out_dict.keys())
            collaborator_weights_unnormalized[col.input] = col.collaborator_task_weight
            times_per_collaborator[col.input] = col.times_per_collaborator
        return tensor_keys_per_col

    def _update_best_model(self, round_dice):
        """Update the best model if the current round's dice score is better."""
        if self.best_dice < round_dice:
            self.best_dice = round_dice
            if self.current_round == 0:
                logger.info(f'Skipping best model saving to disk as it is a random initialization.')
            elif not os.path.exists(f'checkpoint/{self.checkpoint_folder}/temp_model.pkl'):
                raise ValueError(f'Expected temporary model at: checkpoint/{self.checkpoint_folder}/temp_model.pkl to exist but it was not found.')
            else:
                shutil.copyfile(
                    src=f'checkpoint/{self.checkpoint_folder}/temp_model.pkl',
                    dst=f'checkpoint/{self.checkpoint_folder}/best_model.pkl'
                )
                logger.info(f'Saved model with best average binary DICE: {self.best_dice} to checkpoint/{self.checkpoint_folder}/best_model.pkl')


    def _update_aggregator_model(self, inputs):
        """Update the aggregator model with the aggregated tensors."""
        logger.info(f'Aggregator Model updated for round {self.current_round}')
        self.fets_model.model = inputs[0].fets_model.model
        self.fets_model.optimizer = inputs[0].fets_model.optimizer
        self.fets_model.scheduler = inputs[0].fets_model.scheduler
        self.fets_model.params = inputs[0].fets_model.params

        # Rebuild the model with the aggregated tensor_dict
        local_tensor_dict = self._get_aggregated_dict_with_tensorname(self.agg_tensor_dict, self.current_round)
        self.fets_model.rebuild_model(local_tensor_dict)
        self.fets_model.save_native(f'checkpoint/{self.checkpoint_folder}/temp_model.pkl')

    @aggregator
    def start(self):
        # Update experiment results if validation with Hausdorff is included
        if self.include_validation_with_hausdorff:
            self.experiment_results.update({
                f'hausdorff95_label_{label}': [] for label in [0, 1, 2, 4]
            })

        # Initialize the aggregator model
        self._initialize_aggregator_model()

        self.collaborators = self.runtime.collaborators
        # Handle checkpoint restoration or setup a new experiment folder
        if self.restore_from_checkpoint_folder:
            self._restore_from_checkpoint()
        else:
            self._setup_new_experiment()

        # Check if the experiment is already completed
        if self.current_round >= self.n_rounds:
            logger.info("Experiment already completed. Exiting...")
            self.next(self.internal_loop)
            return
        
        if self.restore_from_checkpoint_folder:
            self.current_round += 1

        # Proceed to the next step
        self.collaborator_time_stats = gen_collaborator_time_stats(self.collaborator_names)
        self.next(self.fetch_parameters_for_colls)
    
    @aggregator
    def fetch_parameters_for_colls(self):
        print("*" * 40)
        print("Starting round  {}".format(self.current_round))
        print("*" * 40)
        hparams = self.training_hyper_parameters_for_round(self.collaborators,
                                                            None,
                                                            self.current_round,
                                                            self.collaborators_chosen_each_round,
                                                            self.collaborator_times_per_round)

        learning_rate, epochs_per_round = hparams

        if (epochs_per_round is None):
            logger.warning('Hyper-parameter function warning: function returned None for "epochs_per_round". Setting "epochs_per_round" to 1')
            epochs_per_round = 1
        
        self.hparam_dict = {}
        self.hparam_dict['learning_rate'] = learning_rate
        self.hparam_dict['epochs_per_round'] = epochs_per_round

        logger.info(f'Hyperparameters for round {self.current_round}: {self.hparam_dict}')

        # pick collaborators to train for the round
        self.training_collaborators = self.choose_training_collaborators(self.collaborator_names,
                                                                        None,
                                                                        self.current_round,
                                                                        self.collaborators_chosen_each_round,
                                                                        self.collaborator_times_per_round)
        
        logger.info('Collaborators chosen to train for round {}:\n\t{}'.format(self.current_round, self.training_collaborators))
        self.collaborators_chosen_each_round[self.current_round] = self.training_collaborators

        # Fetch the aggregated tensor dict for the current round
        self.input_tensor_dict = self._get_aggregated_dict_with_tensorname(self.agg_tensor_dict, self.current_round)
        if self.current_round == 0 or self.restored is True:
            self.next(self.initialize_colls, foreach='collaborators')
            self.restored = False
        else:
            self.next(self.aggregated_model_validation, foreach='training_collaborators')

    @collaborator
    def initialize_colls(self):
        if not self.include_validation_with_hausdorff:
            self.gandlf_config['metrics'] = ['dice','dice_per_label']

        logger.info(f'Initializing collaborator {self.input}')
        (
            model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
            params,
        ) = create_pytorch_objects(
            self.gandlf_config, train_csv=self.train_csv_path, val_csv=self.val_csv_path, device=self.device
        )
    
        self.fets_model.device = self.device
        self.fets_model.model = model
        self.fets_model.optimizer = optimizer
        self.fets_model.scheduler = scheduler
        self.fets_model.params = params
        logger.info(f'Initializing dataloaders for collaborator {self.input}')
        collaborator_data_loaders[self.input] = FeTSDataLoader(train_loader, val_loader)

        self.times_per_collaborator = compute_times_per_collaborator(self.input,
                                                                    self.training_collaborators,
                                                                    self.hparam_dict['epochs_per_round'],
                                                                    collaborator_data_loaders[self.input],
                                                                    self.collaborator_time_stats,
                                                                    self.current_round)

        # [TODO] - FIX using Pretrained model
        if self.use_pretrained_model:
            if self.device == 'cpu':
                checkpoint = torch.load(f'checkpoint/pretrained_model/resunet_pretrained.pth',map_location=torch.device('cpu'))
                self.fets_model.model.load_state_dict(checkpoint['model_state_dict'])
                self.fets_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                checkpoint = torch.load(f'checkpoint/pretrained_model/resunet_pretrained.pth')
                self.fets_model.model.load_state_dict(checkpoint['model_state_dict'])
                self.fets_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.next(self.aggregated_model_validation)

    @collaborator
    def aggregated_model_validation(self):
        logger.info(f'Performing aggregated model validation for collaborator {self.input}')
        input_tensor_dict = deepcopy(self.input_tensor_dict)
        val_loader = collaborator_data_loaders[self.input].get_valid_loader()
        self.agg_valid_dict, _ = self.fets_model.validate(self.input, self.current_round, input_tensor_dict, val_loader, apply="global")
        self.next(self.train)

    @collaborator
    def train(self):
        logger.info(f'Performing training for collaborator {self.input}')
        train_loader = collaborator_data_loaders[self.input].get_train_loader()
        input_tensor_dict = deepcopy(self.input_tensor_dict)
        self.global_output_tensor_dict, _ =  self.fets_model.train(self.input, self.current_round, input_tensor_dict, self.hparam_dict, train_loader)
        self.collaborator_task_weight = collaborator_data_loaders[self.input].get_train_data_size()
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):        
        logger.info(f'Performing local model validation for collaborator {self.input}')
        val_loader = collaborator_data_loaders[self.input].get_valid_loader()
        # Update the model with the trained tensors for local validation of this round.
        input_tensor_dict = self._get_aggregated_dict_with_tensorname(self.global_output_tensor_dict, self.current_round, 'trained')
        self.local_valid_dict, _ = self.fets_model.validate(self.input, self.current_round, input_tensor_dict, val_loader, apply="local")
        self.next(self.join)

    @aggregator
    def join(self, inputs):
        logger.info(f'Aggregating results for round {self.current_round}')
        agg_tensor_db = TensorDB() # Used for aggregating and persisting tensors
        collaborator_weights_unnormalized = {}
        times_per_collaborator = {}
        tensor_keys_per_col = ()

        # Cache the aggregator tensor dict in tensor_db so that tensor_db has updated tensor values.
        agg_tensor_db.cache_tensor(self.agg_tensor_dict)
        
        # Process each collaborator's tensors
        tensor_keys_per_col = self._process_collaborators(inputs, agg_tensor_db, collaborator_weights_unnormalized, times_per_collaborator)

        self.collaborator_times_per_round[self.current_round] = times_per_collaborator
        weight_total = sum(collaborator_weights_unnormalized.values())
        collaborator_weight_dict = {
            k: v / weight_total for k, v in collaborator_weights_unnormalized.items()
        }
        logger.info(f'Calculated Collaborator weights: {collaborator_weight_dict} and and times: {times_per_collaborator}')

        # Perform aggregation
        self._aggregate_tensors(agg_tensor_db, tensor_keys_per_col, collaborator_weight_dict)

        # Clean up the tensor_db for the round_data_to_delete rounds
        agg_tensor_db.clean_up(self.db_store_rounds)

        times_list = [(t, col) for col, t in times_per_collaborator.items()]
        times_list = sorted(times_list)

        # the round time is the max of the times_list
        round_time = max([t for t, _ in times_list])
        self.total_simulated_time += round_time

        ## CONVERGENCE METRIC COMPUTATION
        # update the auc score
        self.best_dice_over_time_auc += self.best_dice * round_time

        # project the auc score as remaining time * best dice
        # this projection assumes that the current best score is carried forward for the entire week
        projected_auc = (MAX_SIMULATION_TIME - self.total_simulated_time) * self.best_dice + self.best_dice_over_time_auc
        projected_auc /= MAX_SIMULATION_TIME

        # update metrics and results
        summary, round_dice = self._update_metrics(
            self.current_round, agg_tensor_db, self.experiment_results,
            self.include_validation_with_hausdorff, self.total_simulated_time, projected_auc
        )

        # Update the best model if necessary
        self._update_best_model(round_dice)

        # Update the agg_tensor_dict for subsequent rounds with the aggregated tensor_db
        self.agg_tensor_dict.clear()
        self.agg_tensor_dict = {
            TensorKey(record["tensor_name"], record["origin"], record["round"], record["report"], record["tags"]): record["nparray"]
            for _, record in agg_tensor_db.tensor_db.iterrows()
        }

        if self.save_checkpoints:
            logger.info(f'Saving checkpoint for round {self.current_round} : checkpoint folder {self.checkpoint_folder}')
            logger.info(f'To resume from this checkpoint, set the restore_from_checkpoint_folder parameter to {self.checkpoint_folder}')
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
        if self.total_simulated_time > MAX_SIMULATION_TIME:
            logger.info("Simulation time exceeded. Ending Experiment")
            self.next(self.end)
            return

        # Update the aggregator model and rebuild it with aggregated tensors
        self._update_aggregator_model(inputs)
        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        if self.current_round >= self.n_rounds:
            print('************* EXPERIMENT COMPLETED *************')
            print('Experiment results:')
            print(pd.DataFrame.from_dict(self.experiment_results))
            self.next(self.end)
        else:
            self.current_round += 1
            self.next(self.fetch_parameters_for_colls)

    @aggregator
    def end(self):
        logger.info('********************************')
        logger.info('End of flow')
        logger.info('********************************')
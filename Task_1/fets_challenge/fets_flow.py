import os
import shutil
import time
import logging
from copy import deepcopy
import pandas as pd
from pathlib import Path

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from openfl.databases import TensorDB
from openfl.utilities import TensorKey, change_tags

from .fets_data_loader import FeTSDataLoader

from .checkpoint_utils import setup_checkpoint_folder, save_checkpoint, load_checkpoint
from .time_utils import gen_collaborator_time_stats, compute_times_per_collaborator, MAX_SIMULATION_TIME

from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_metric(metric_name, fl_round, agg_tensor_db):
    target_tags = ('metric', 'validate_agg')
    metric_tensor_key = TensorKey(metric_name, 'aggregator', fl_round, True, target_tags)
    nparray = agg_tensor_db.get_tensor_from_cache(metric_tensor_key)
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

def return_cleanup_key(tensor_key, col, round_data_to_delete):
    new_tags = tensor_key.tags
    if col in tensor_key.tags:
        new_tags = change_tags(tensor_key.tags, remove_field=col)
    modified_key = TensorKey(
        tensor_name=tensor_key.tensor_name,
        origin=col,
        round_number=round_data_to_delete,
        report=tensor_key.report,
        tags=new_tags
    )
    return modified_key

def get_aggregated_dict_with_tensorname(agg_tensor_dict):
    agg_dict_with_tensornames = {}
    for tensor_key, value in agg_tensor_dict.items():
        tensor_name, origin, round_number, report, tags = tensor_key
        agg_dict_with_tensornames[tensor_name] = value
    return agg_dict_with_tensornames

def update_metrics(current_round, agg_tensor_db, summary, experiment_results, include_validation_with_hausdorff, 
                   total_simulated_time, round_dice, projected_auc):

        round_loss = get_metric('valid_loss', current_round, agg_tensor_db)
        round_dice = get_metric('valid_dice', current_round, agg_tensor_db)
        dice_label_0 = get_metric('valid_dice_per_label_0', current_round, agg_tensor_db)
        dice_label_1 = get_metric('valid_dice_per_label_1', current_round, agg_tensor_db)
        dice_label_2 = get_metric('valid_dice_per_label_2', current_round, agg_tensor_db)
        dice_label_4 = get_metric('valid_dice_per_label_4', current_round, agg_tensor_db)
        if include_validation_with_hausdorff:
            hausdorff95_label_0 = get_metric('valid_hd95_per_label_0', current_round, agg_tensor_db)
            hausdorff95_label_1 = get_metric('valid_hd95_per_label_1', current_round, agg_tensor_db)
            hausdorff95_label_2 = get_metric('valid_hd95_per_label_2', current_round, agg_tensor_db)
            hausdorff95_label_4 = get_metric('valid_hd95_per_label_4', current_round, agg_tensor_db)
        
        # # End of round summary
        summary = '"**** END OF ROUND {} SUMMARY *****"'.format(current_round)
        summary += "\n\tSimulation Time: {} minutes".format(round(total_simulated_time / 60, 2))
        summary += "\n\t(Projected) Convergence Score: {}".format(projected_auc)
        summary += "\n\tRound Loss: {}".format(round_loss)
        summary += "\n\tRound Dice: {}".format(round_dice)
        summary += "\n\tDICE Label 0: {}".format(dice_label_0)
        summary += "\n\tDICE Label 1: {}".format(dice_label_1)
        summary += "\n\tDICE Label 2: {}".format(dice_label_2)
        summary += "\n\tDICE Label 4: {}".format(dice_label_4)
        if include_validation_with_hausdorff:
            summary += "\n\tHausdorff95 Label 0: {}".format(hausdorff95_label_0)
            summary += "\n\tHausdorff95 Label 1: {}".format(hausdorff95_label_1)
            summary += "\n\tHausdorff95 Label 2: {}".format(hausdorff95_label_2)
            summary += "\n\tHausdorff95 Label 4: {}".format(hausdorff95_label_4)
        logger.info(summary)

        experiment_results['round'].append(current_round)
        experiment_results['time'].append(total_simulated_time)
        experiment_results['convergence_score'].append(projected_auc)
        experiment_results['round_dice'].append(round_dice)
        experiment_results['dice_label_0'].append(dice_label_0)
        experiment_results['dice_label_1'].append(dice_label_1)
        experiment_results['dice_label_2'].append(dice_label_2)
        experiment_results['dice_label_4'].append(dice_label_4)
        if include_validation_with_hausdorff:
            experiment_results['hausdorff95_label_0'].append(hausdorff95_label_0)
            experiment_results['hausdorff95_label_1'].append(hausdorff95_label_1)
            experiment_results['hausdorff95_label_2'].append(hausdorff95_label_2)
            experiment_results['hausdorff95_label_4'].append(hausdorff95_label_4)

collaborator_data_loaders = {}

class FeTSFederatedFlow(FLSpec):
    def __init__(self, fets_model, params_dict, rounds=5 , device="cpu",  **kwargs):
        super().__init__(**kwargs)
        self.fets_model = fets_model
        self.n_rounds = rounds
        self.device = device
        self.current_round = 1
        self.total_simulated_time = 0
        self.best_dice = -1.0
        self.best_dice_over_time_auc = 0
        self.collaborators_chosen_each_round = {}
        self.collaborator_times_per_round = {}
        self.agg_tensor_dict = {}
        self.tensor_keys_per_col = {}
        self.restored = False

        self.include_validation_with_hausdorff = params_dict.get('include_validation_with_hausdorff', False)
        self.choose_training_collaborators = params_dict.get('choose_training_collaborators', None)
        self.training_hyper_parameters_for_round = params_dict.get('training_hyper_parameters_for_round', None)
        self.restore_from_checkpoint_folder = params_dict.get('restore_from_checkpoint_folder', None)
        self.save_checkpoints = params_dict.get('save_checkpoints', False)

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

    @aggregator
    def start(self):

        if self.include_validation_with_hausdorff:
            self.experiment_results.update({
                'hausdorff95_label_0': [],
                'hausdorff95_label_1': [],
                'hausdorff95_label_2': [],
                'hausdorff95_label_4': [],
            })

        self.collaborators = self.runtime.collaborators
        if self.restore_from_checkpoint_folder is None:
            self.checkpoint_folder = setup_checkpoint_folder()
            logger.info(f'\nCreated experiment folder {self.checkpoint_folder}...')
            starting_round_num = 0
        else:
            if not Path(f'checkpoint/{self.restore_from_checkpoint_folder}').exists():
                logger.warning(f'Could not find provided checkpoint folder: {self.restore_from_checkpoint_folder}. Exiting...')
                exit(1)
            else:
                #TODO : Validate load from checkpoint logic
                logger.info(f'Attempting to load last completed round from {self.restore_from_checkpoint_folder}')
                state = load_checkpoint(self.restore_from_checkpoint_folder)
                self.checkpoint_folder = self.restore_from_checkpoint_folder

                [loaded_collaborator_names, starting_round_num, self.collaborator_time_stats,
                self.total_simulated_time, self.best_dice, self.best_dice_over_time_auc,
                self.collaborators_chosen_each_round, self.collaborator_times_per_round,
                self.tensor_keys_per_col, self.experiment_results, summary, agg_tensor_db] = state

                if loaded_collaborator_names != self.collaborator_names:
                    logger.error(f'Collaborator names found in checkpoint ({loaded_collaborator_names}) '
                                f'do not match provided collaborators ({self.collaborator_names})')
                    exit(1)

                self.restored = True
                logger.info(f'Previous summary for round {starting_round_num}')
                logger.info(summary)

                aggregator_tensor_db = TensorDB()
                aggregator_tensor_db.tensor_db = agg_tensor_db

                #Updating the agg_tensor_dict from stored tensor_db
                starting_round_num += 1
                self.current_round = starting_round_num
                logger.info(f'Loading checkpoint from round {self.tensor_keys_per_col}')
                for col,tensor_keys in self.tensor_keys_per_col.items():
                    for tensor_key in tensor_keys:
                        tensor_name, _, _, _, _ = tensor_key
                        if tensor_name not in self.agg_tensor_dict:
                            self.agg_tensor_dict[tensor_key] = aggregator_tensor_db.get_tensor_from_cache(tensor_key)
                            logger.info(f'Loaded tensor key {tensor_key}')

        if self.current_round >= self.n_rounds:
            logger.info("Experiment already completed. Exiting...")
            self.next(self.end)
                        
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
        if self.current_round == 1 or self.restored is True:
            self.next(self.initialize_colls, foreach='collaborators')
            self.restored = False
        else:
            self.next(self.aggregated_model_validation, foreach='training_collaborators')

    @collaborator
    def initialize_colls(self):

        gandlf_conf = {}
        if isinstance(self.gandlf_config, str) and os.path.exists(self.gandlf_config):
            gandlf_conf = ConfigManager(self.gandlf_config)
        elif isinstance(self.gandlf_config, dict):
            gandlf_conf = self.gandlf_config
        else:
            exit("GANDLF config file not found. Exiting...")

        if not self.include_validation_with_hausdorff:
            gandlf_conf['metrics'] = ['dice','dice_per_label']

        logger.info(f'Initializing collaborator {self.input}')
        (
            model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
            params,
        ) = create_pytorch_objects(
            gandlf_conf, train_csv=self.train_csv_path, val_csv=self.val_csv_path, device=self.device
        )
    
        self.fets_model.device = self.device
        self.fets_model.model = model
        self.fets_model.optimizer = optimizer
        self.fets_model.scheduler = scheduler
        self.fets_model.params = params
        logger.info(f'Initializing dataloaders for collaborator {self.input}')
        collaborator_data_loaders[self.input] = FeTSDataLoader(train_loader, val_loader)


        #TODO Validate the times per collaborator is calculated based on the random values, it doesn't look like the actual time taken by the collaborator 
        self.times_per_collaborator = compute_times_per_collaborator(self.input,
                                                                    self.training_collaborators,
                                                                    self.hparam_dict['epochs_per_round'],
                                                                    collaborator_data_loaders[self.input],
                                                                    self.collaborator_time_stats,
                                                                    self.current_round)

        if self.restored is False:
            tensor_dict = self.fets_model.get_tensor_dict()
            for key, value in tensor_dict.items():
                origin = 'collaborator'
                round_number = self.current_round
                report = False
                tags = ('trained')
                agg_tensor_key = TensorKey(key, origin, round_number, report, tags)

                self.agg_tensor_dict[agg_tensor_key] = value
        self.next(self.aggregated_model_validation)

    @collaborator
    def aggregated_model_validation(self):
        logger.info(f'Performing aggregated model validation for collaborator {self.input}')
        input_tensor_dict = get_aggregated_dict_with_tensorname(self.agg_tensor_dict)
        val_loader = collaborator_data_loaders[self.input].get_valid_loader()
        self.fets_model.rebuild_model(self.current_round, input_tensor_dict)
        self.agg_valid_dict, _ = self.fets_model.validate(self.input, self.current_round, val_loader, apply="global")
        self.next(self.train)

    @collaborator
    def train(self):
        logger.info(f'Performing training for collaborator {self.input}')
        train_loader = collaborator_data_loaders[self.input].get_train_loader()
        self.global_output_tensor_dict, _ =  self.fets_model.train(self.input, self.current_round, self.hparam_dict, train_loader)
        self.collaborator_task_weight = collaborator_data_loaders[self.input].get_train_data_size()
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):        
        logger.info(f'Performing local model validation for collaborator {self.input}')
        val_loader = collaborator_data_loaders[self.input].get_valid_loader()
        self.local_valid_dict, _ = self.fets_model.validate(self.input, self.current_round, val_loader, apply="local")
        self.next(self.join)

    @aggregator
    def join_task(self, inputs):
        self.next(self.internal_loop)

    @aggregator
    def join(self, inputs):
        round_data_to_delete = 0
        if self.current_round > self.db_store_rounds:
            round_data_to_delete = self.current_round - self.db_store_rounds
        self.aggregation_type.set_state_data_for_round(self.collaborators_chosen_each_round, self.collaborator_times_per_round)
        agg_tensor_db = TensorDB()
        collaborator_weights_unnormalized = {}
        times_per_collaborator = {}
        for idx, col in enumerate(inputs):
            logger.info(f'Aggregating results for {idx}')
            agg_out_dict = {}
            cache_tensor_dict(col.local_valid_dict, agg_tensor_db, idx, agg_out_dict)
            cache_tensor_dict(col.agg_valid_dict, agg_tensor_db, idx, agg_out_dict)
            cache_tensor_dict(col.global_output_tensor_dict, agg_tensor_db, idx, agg_out_dict)            
            self.agg_tensor_dict.update(agg_out_dict)
            
            # Store the keys for each collaborator
            self.tensor_keys_per_col[str(idx + 1)] = list(agg_out_dict.keys())
            
            #TODO : Compare the weight from the old expermient, we saw three different sets of weights while running the experiment for single round
            collaborator_weights_unnormalized[col.input] = col.collaborator_task_weight
            times_per_collaborator[col.input] = col.times_per_collaborator

        weight_total = sum(collaborator_weights_unnormalized.values())
        collaborator_weight_dict = {
            k: v / weight_total for k, v in collaborator_weights_unnormalized.items()
        }
        logger.info(f'Calculated Collaborator weights: {collaborator_weight_dict} and and times: {times_per_collaborator}')
        agg_tensor_keys = []
        for col,tensor_keys in self.tensor_keys_per_col.items():
            for tensor_key in tensor_keys:
                tensor_name, origin, round_number, report, tags = tensor_key
                if col in tags:
                    new_tags = change_tags(tags, remove_field=col)
                    agg_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
                    # Aggregates the tensor values for the tensor key and stores it in tensor_db
                    if agg_tensor_key not in self.agg_tensor_dict:
                        agg_results = agg_tensor_db.get_aggregated_tensor(
                            agg_tensor_key,
                            collaborator_weight_dict,
                            aggregation_function=self.aggregation_type,
                        )
                        self.agg_tensor_dict[agg_tensor_key] = agg_tensor_db.get_tensor_from_cache(agg_tensor_key)
                        agg_tensor_keys.append(agg_tensor_key)

                #cleaningup aggregated tensor dict based on db store rounds
                if self.current_round > self.db_store_rounds:
                    col_tensor_key_to_be_deleted = return_cleanup_key(tensor_key, col, round_data_to_delete)
                    agg_tensor_key_to_be_deleted = TensorKey(tensor_name, origin, round_data_to_delete, report, new_tags)
                    if col_tensor_key_to_be_deleted in self.agg_tensor_dict:
                        self.agg_tensor_dict.pop(col_tensor_key_to_be_deleted)
                    if agg_tensor_key_to_be_deleted in self.agg_tensor_dict:
                        self.agg_tensor_dict.pop(agg_tensor_key_to_be_deleted)

        self.tensor_keys_per_col['aggregator'] = agg_tensor_keys

        for key in self.agg_tensor_dict.keys():
            print(f'[Kush Aggregated Tensor Dictionary] Keys : {key}')

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
        summary = ""
        round_dice = 0
        update_metrics(self.current_round, agg_tensor_db, summary, self.experiment_results, 
                        self.include_validation_with_hausdorff, self.total_simulated_time, round_dice, projected_auc)

        if self.best_dice < round_dice:
            self.best_dice = round_dice
            # Set the weights for the final model
            if self.current_round == 1:
                # here the initial model was validated (temp model does not exist)
                logger.info(f'Skipping best model saving to disk as it is a random initialization.')
            elif not os.path.exists(f'checkpoint/{self.checkpoint_folder}/temp_model.pkl'):
                raise ValueError(f'Expected temporary model at: checkpoint/{self.checkpoint_folder}/temp_model.pkl to exist but it was not found.')
            else:
                # here the temp model was the one validated
                shutil.copyfile(src=f'checkpoint/{self.checkpoint_folder}/temp_model.pkl',dst=f'checkpoint/{self.checkpoint_folder}/best_model.pkl')
                logger.info(f'Saved model with best average binary DICE: {self.best_dice} to checkpoint/{self.checkpoint_folder}/best_model.pkl')

        # cache the aggregated tensor_dict
        cache_tensor_dict(self.agg_tensor_dict, agg_tensor_db, 0, {})

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
                            self.tensor_keys_per_col,
                            self.experiment_results,
                            summary)

        # if the total_simulated_time has exceeded the maximum time, we break
        # in practice, this means that the previous round's model is the last model scored,
        # so a long final round should not actually benefit the competitor, since that final
        # model is never globally validated
        if self.total_simulated_time > MAX_SIMULATION_TIME:
            logger.info("Simulation time exceeded. Ending Experiment")
            self.next(self.end)

        # save the most recent aggregated model in native format to be copied over as best when appropriate
        # (note this model has not been validated by the collaborators yet)
        # Global FeTS Model may be unititialized in the first round
        if self.fets_model.model is None:
            logger.info(f'Global model is not initialized. Initializing with the first round model')
            self.fets_model.model = inputs[0].fets_model.model
            self.fets_model.optimizer = inputs[0].fets_model.optimizer
            self.fets_model.scheduler = inputs[0].fets_model.scheduler
            self.fets_model.params = inputs[0].fets_model.params

        # Rebuild the model with the aggregated tensor_dict
        local_tensor_dict = get_aggregated_dict_with_tensorname(self.agg_tensor_dict)
        self.fets_model.rebuild_model(self.current_round, local_tensor_dict)
        self.fets_model.save_native(f'checkpoint/{self.checkpoint_folder}/temp_model.pkl')
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
# Provided by the FeTS Initiative (www.fets.ai) as part of the FeTS Challenge 2022

# Contributing Authors (alphabetical):
# Patrick Foley (Intel), Micah Sheller (Intel)

import os
import warnings
from collections import namedtuple
from copy import copy
import shutil
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from openfl.utilities.split import split_tensor_dict_for_holdouts
from openfl.utilities import TensorKey
from openfl.protocols import utils
import openfl.native as fx
from openfl.databases import TensorDB
import torch

from .gandlf_csv_adapter import construct_fedsim_csv, extract_csv_partitions
from .custom_aggregation_wrapper import CustomAggregationWrapper
from .checkpoint_utils import setup_checkpoint_folder, save_checkpoint, load_checkpoint

from .fets_flow import FeTSFederatedFlow
from .fets_challenge_model import FeTSChallengeModel

from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime

logger = getLogger(__name__)
# This catches PyTorch UserWarnings for CPU
warnings.filterwarnings("ignore", category=UserWarning)

# one week
# MINUTE = 60
# HOUR = 60 * MINUTE
# DAY = 24 * HOUR
# WEEK = 7 * DAY
MAX_SIMULATION_TIME = 7 * 24 * 60 * 60 

def aggregator_private_attributes(
       uuid, aggregation_type, collaborator_names, include_validation_with_hausdorff, choose_training_collaborators, 
       training_hyper_parameters_for_round, restore_from_checkpoint_folder, save_checkpoints):
    return {"uuid": uuid,
            "aggregation_type" : aggregation_type,
            "collaborator_names": collaborator_names,
            "include_validation_with_hausdorff": include_validation_with_hausdorff,
            "choose_training_collaborators": choose_training_collaborators,
            "training_hyper_parameters_for_round": training_hyper_parameters_for_round,
            "max_simulation_time": MAX_SIMULATION_TIME,
            "restore_from_checkpoint_folder": restore_from_checkpoint_folder,
            "save_checkpoints":save_checkpoints
}
 

def collaborator_private_attributes(
        index, n_collaborators, gandlf_config, train_csv_path, val_csv_path):
        return {
            "index": index,
            "n_collaborators": n_collaborators,
            "gandlf_config": gandlf_config,
            "train_csv_path": train_csv_path,
            "val_csv_path": val_csv_path
        }


def run_challenge_experiment(aggregation_function,
                             choose_training_collaborators,
                             training_hyper_parameters_for_round,
                             institution_split_csv_filename,
                             brats_training_data_parent_dir,
                             db_store_rounds=5,
                             rounds_to_train=5,
                             device='cpu',
                             save_checkpoints=True,
                             restore_from_checkpoint_folder=None, 
                             include_validation_with_hausdorff=True,
                             use_pretrained_model=False):

    from sys import path, exit

    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()

    gandlf_config_path = os.path.join(root, 'gandlf_config.yaml')
    path.append(str(root))
    path.insert(0, str(work))
    
    # create gandlf_csv and get collaborator names
    gandlf_csv_path = os.path.join(work, 'gandlf_paths.csv')
    # split_csv_path = os.path.join(work, institution_split_csv_filename)
    collaborator_names = construct_fedsim_csv(brats_training_data_parent_dir,
                                              institution_split_csv_filename,
                                              0.8,
                                              gandlf_csv_path)
    
    print(f'Collaborator names for experiment : {collaborator_names}')

    aggregation_wrapper = CustomAggregationWrapper(aggregation_function) # ---> [TODO] Set the aggregation function in the workflow

    # [TODO] [Workflow - API] Need to check db_store rounds
    # overrides = {
    #     'aggregator.settings.db_store_rounds': db_store_rounds,
    # }

    # [TODO] [Workflow - API] How to update the gandfl_config runtime
    # if not include_validation_with_hausdorff:
    #     plan.config['task_runner']['settings']['fets_config_dict']['metrics'] = ['dice','dice_per_label']

    transformed_csv_dict = extract_csv_partitions(os.path.join(work, 'gandlf_paths.csv'))

    collaborators = []
    for idx, col in enumerate(collaborator_names):
        col_dir = os.path.join(work, 'data', str(col))
        os.makedirs(col_dir, exist_ok=True)

        train_csv_path = os.path.join(col_dir, 'train.csv')
        val_csv_path = os.path.join(col_dir, 'valid.csv')

        transformed_csv_dict[col]['train'].to_csv(train_csv_path)
        transformed_csv_dict[col]['val'].to_csv(val_csv_path)

        collaborators.append(
            Collaborator(
                name=col,
                private_attributes_callable=collaborator_private_attributes,
                # If 1 GPU is available in the machine
                # Set `num_gpus=0.0` to `num_gpus=0.3` to run on GPU
                # with ray backend with 2 collaborators
                num_cpus=4.0,
                num_gpus=0.0,
                # arguments required to pass to callable
                index=idx,
                n_collaborators=len(collaborator_names),
                gandlf_config=gandlf_config_path,
                train_csv_path=train_csv_path,
                val_csv_path=val_csv_path
            )
        )

    aggregator = Aggregator(name="aggregator",
                            private_attributes_callable=aggregator_private_attributes,
                            num_cpus=4.0,
                            num_gpus=0.0,
                            uuid='aggregator',
                            collaborator_names=collaborator_names,
                            include_validation_with_hausdorff=include_validation_with_hausdorff,
                            aggregation_type=aggregation_wrapper,
                            choose_training_collaborators=choose_training_collaborators,
                            training_hyper_parameters_for_round=training_hyper_parameters_for_round,
                            restore_from_checkpoint_folder=restore_from_checkpoint_folder,
                            save_checkpoints=save_checkpoints)

    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators, backend="single_process", num_actors=1
    )

    logger.info(f"Local runtime collaborators = {local_runtime.collaborators}")

    model = FeTSChallengeModel(gandlf_config_path)
    flflow = FeTSFederatedFlow(
        model,
        rounds_to_train,
        device,
    )

    flflow.runtime = local_runtime
    flflow.run()

    # [TODO] [Workflow - API] -> Commenting as pretrained model is not used.
    # ---> Define a new step in federated flow before training to load the pretrained model
    # if use_pretrained_model:
    #     print('TESTING ->>>>>> Loading pretrained model...')
    #     if device == 'cpu':
    #         checkpoint = torch.load(f'{root}/pretrained_model/resunet_pretrained.pth',map_location=torch.device('cpu'))
    #         print('TESTING ->>>>>> Loading checkpoint model...')
    #         print(checkpoint.keys())
    #         print('TESTING ->>>>>> Loading checkpoint state dict...')
    #         model_state = checkpoint['model_state_dict']
    #         for name, tensor in model_state.items():
    #             print(f"Priting {name}: {tensor.shape}")
    #         print('TESTING ->>>>>> Loading taskrunner model')
    #         print(task_runner.model)    
    #         task_runner.model.load_state_dict(checkpoint['model_state_dict'])
    #         task_runner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     else:
    #         checkpoint = torch.load(f'{root}/pretrained_model/resunet_pretrained.pth')
    #         task_runner.model.load_state_dict(checkpoint['model_state_dict'])
    #         task_runner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # # Initialize model weights
    # # [TODO] [Workflow - API] How to set the initial state in the workflow
    # init_state_path = plan.config['aggregator']['settings']['init_state_path']
    # tensor_dict, _ = split_tensor_dict_for_holdouts(logger, task_runner.get_tensor_dict(False))

    # model_snap = utils.construct_model_proto(tensor_dict=tensor_dict,
    #                                          round_number=0,
    #                                          tensor_pipe=tensor_pipe)

    # utils.dump_proto(model_proto=model_snap, fpath=init_state_path)

    #return pd.DataFrame.from_dict(experiment_results), checkpoint_folder
    return None
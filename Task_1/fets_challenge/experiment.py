# Provided by the FeTS Initiative (www.fets.ai) as part of the FeTS Challenge 2022

# Contributing Authors (alphabetical):
# Patrick Foley (Intel), Micah Sheller (Intel)

import os
from copy import deepcopy
import warnings
from logging import getLogger
from pathlib import Path
from torch.utils.data import DataLoader

from .gandlf_csv_adapter import construct_fedsim_csv, extract_csv_partitions
from .custom_aggregation_wrapper import CustomAggregationWrapper

from .fets_flow import FeTSFederatedFlow
from .fets_challenge_model import FeTSChallengeModel
from .fets_data_loader import FeTSDataLoader

from openfl.experimental.workflow.interface import Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime

from GANDLF.config_manager import ConfigManager

logger = getLogger(__name__)
# This catches PyTorch UserWarnings for CPU
warnings.filterwarnings("ignore", category=UserWarning)

def aggregator_private_attributes(aggregation_type, collaborator_names, db_store_rounds):
    return {
        "aggregation_type" : aggregation_type,
        "collaborator_names": collaborator_names,
        "checkpoint_folder":None,
        "db_store_rounds":db_store_rounds,
        "agg_tensor_dict":{}
    }


def collaborator_private_attributes(index, train_csv_path, val_csv_path):
    return {
        "index": index,
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
                             use_pretrained_model=False,
                             backend_process='single_process'):

    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()
    gandlf_config_path = os.path.join(root, 'config', 'gandlf_config.yaml')
    
    # create gandlf_csv and get collaborator names
    gandlf_csv_path = os.path.join(work, 'gandlf_paths.csv')
    split_csv_path = os.path.join(work, institution_split_csv_filename)
    collaborator_names = construct_fedsim_csv(brats_training_data_parent_dir,
                                              split_csv_path,
                                              0.8,
                                              gandlf_csv_path)
    
    logger.info(f'Collaborator names for experiment : {collaborator_names}')

    aggregation_wrapper = CustomAggregationWrapper(aggregation_function)

    transformed_csv_dict = extract_csv_partitions(os.path.join(work, 'gandlf_paths.csv'))

    gandlf_conf = {}
    if isinstance(gandlf_config_path, str) and os.path.exists(gandlf_config_path):
        gandlf_conf = ConfigManager(gandlf_config_path)
    elif isinstance(gandlf_config_path, dict):
        gandlf_conf = gandlf_config_path
    else:
        exit("GANDLF config file not found. Exiting...")

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
                # private arguments required to pass to callable
                index=idx,
                train_csv_path=train_csv_path,
                val_csv_path=val_csv_path
            )
        )

    aggregator = Aggregator(name="aggregator",
                            private_attributes_callable=aggregator_private_attributes,
                            num_cpus=4.0,
                            num_gpus=0.0,
                            # private arguments required to pass to callable
                            collaborator_names=collaborator_names,
                            aggregation_type=aggregation_wrapper,
                            db_store_rounds=db_store_rounds)

    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators, backend=backend_process, num_actors=1
    )

    logger.info(f"Local runtime collaborators = {local_runtime.collaborators}")

    params_dict = {"include_validation_with_hausdorff": include_validation_with_hausdorff,
                   "use_pretrained_model": use_pretrained_model,
                   "gandlf_config": gandlf_conf,
                    "choose_training_collaborators": choose_training_collaborators,
                    "training_hyper_parameters_for_round": training_hyper_parameters_for_round,
                    "restore_from_checkpoint_folder": restore_from_checkpoint_folder,
                    "save_checkpoints": save_checkpoints}

    model = FeTSChallengeModel()
    flflow = FeTSFederatedFlow(
        model,
        params_dict,
        rounds_to_train,
        device
    )

    flflow.runtime = local_runtime
    flflow.run()
    return aggregator.private_attributes["checkpoint_folder"]
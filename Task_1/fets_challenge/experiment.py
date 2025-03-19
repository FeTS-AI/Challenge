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

logger = getLogger(__name__)
# This catches PyTorch UserWarnings for CPU
warnings.filterwarnings("ignore", category=UserWarning)

def aggregator_private_attributes(
       aggregation_type, collaborator_names, db_store_rounds):
    return {"aggregation_type" : aggregation_type,
            "collaborator_names": collaborator_names,
            "checkpoint_folder":None,
            "db_store_rounds":db_store_rounds
}
 

def collaborator_private_attributes(
        index, gandlf_config, train_csv_path, val_csv_path):
        return {
            "index": index,
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
    
    print(f'Collaborator names for experiment : {collaborator_names}')

    aggregation_wrapper = CustomAggregationWrapper(aggregation_function)

    # [TODO] Handle the storing of data in the fets flow (add db_sotre_rounds aggregator private attribute)
    # overrides = {
    #     'aggregator.settings.db_store_rounds': db_store_rounds,
    # }

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
                gandlf_config=gandlf_config_path,
                train_csv_path=train_csv_path,
                val_csv_path=val_csv_path
            )
        )

    aggregator = Aggregator(name="aggregator",
                            private_attributes_callable=aggregator_private_attributes,
                            num_cpus=4.0,
                            num_gpus=0.0,
                            collaborator_names=collaborator_names,
                            aggregation_type=aggregation_wrapper,
                            db_store_rounds=db_store_rounds)

    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators, backend="single_process", num_actors=1
    )

    logger.info(f"Local runtime collaborators = {local_runtime.collaborators}")

    params_dict = {"include_validation_with_hausdorff": include_validation_with_hausdorff,
              "choose_training_collaborators": choose_training_collaborators,  #TODO verify with different collaborators and check if works?
              "training_hyper_parameters_for_round": training_hyper_parameters_for_round,
              "restore_from_checkpoint_folder": restore_from_checkpoint_folder,
              "save_checkpoints": save_checkpoints}

    model = FeTSChallengeModel(gandlf_config_path)
    flflow = FeTSFederatedFlow(
        model,
        params_dict,
        rounds_to_train,
        device
    )

    flflow.runtime = local_runtime
    flflow.run()

    # #TODO [Workflow - API] -> Commenting as pretrained model is not used.
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
    # #TODO [Workflow - API] How to set the initial state in the workflow -> check if it needed to be done in workflow
    # init_state_path = plan.config['aggregator']['settings']['init_state_path']
    # tensor_dict, _ = split_tensor_dict_for_holdouts(logger, task_runner.get_tensor_dict(False))
    # model_snap = utils.construct_model_proto(tensor_dict=tensor_dict,
    #                                          round_number=0,
    #                                          tensor_pipe=tensor_pipe)
    # utils.dump_proto(model_proto=model_snap, fpath=init_state_path)
    return aggregator.private_attributes["checkpoint_folder"]
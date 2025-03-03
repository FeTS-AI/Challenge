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
import torch

from .gandlf_csv_adapter import construct_fedsim_csv, extract_csv_partitions
from .custom_aggregation_wrapper import CustomAggregationWrapper
from .checkpoint_utils import setup_checkpoint_folder, save_checkpoint, load_checkpoint

from .fets_flow import FeTSFederatedFlow
from .fets_challenge_model import FeTSChallengeModel

from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime

# one week
# MINUTE = 60
# HOUR = 60 * MINUTE
# DAY = 24 * HOUR
# WEEK = 7 * DAY
MAX_SIMULATION_TIME = 7 * 24 * 60 * 60 

## COLLABORATOR TIMING DISTRIBUTIONS
# These data are derived from the actual timing information in the real-world FeTS information
# They reflect a subset of the institutions involved.
# Tuples are (mean, stddev) in seconds

# time to train one patient
TRAINING_TIMES = [(6.710741331207654, 0.8726112813698301),
                  (2.7343911917098445, 0.023976155580152165),
                  (3.173076923076923, 0.04154320960517865),
                  (6.580379746835443, 0.22461890673025595),
                  (3.452046783625731, 0.47136389322749656),
                  (6.090788461700995, 0.08541499003440205),
                  (3.206933911159263, 0.1927067498514361),
                  (3.3358208955223883, 0.2950567549663471),
                  (4.391304347826087, 0.37464538999161057),
                  (6.324805129494594, 0.1413885448869165),
                  (7.415133477633478, 1.1198881747151301),
                  (5.806410256410255, 0.029926699295169234),
                  (6.300204918032787, 0.24932319729777577),
                  (5.886317567567567, 0.018627858809133223),
                  (5.478184991273998, 0.04902740607167421),
                  (6.32440159574468, 0.15838847558954935),
                  (20.661918328585003, 6.085405543890793),
                  (3.197901325478645, 0.07049966132127056),
                  (6.523963730569948, 0.2533266757118492),
                  (2.6540077569489338, 0.025503099659276184),
                  (1.8025746183640918, 0.06805805332403576)]

# time to validate one patient
VALIDATION_TIMES = [(23.129135113591072, 2.5975116854269507),
                    (12.965544041450777, 0.3476297824941513),
                    (14.782051282051283, 0.5262660449172765),
                    (16.444936708860762, 0.42613177203005187),
                    (15.728654970760235, 4.327559980390658),
                    (12.946098012884802, 0.2449927822869217),
                    (15.335950126991456, 1.1587597276712558),
                    (24.024875621890544, 3.087348297794285),
                    (38.361702127659576, 2.240113332190875),
                    (16.320970580839827, 0.4995108101783225),
                    (30.805555555555554, 3.1836337269688237),
                    (12.100899742930592, 0.41122386959584895),
                    (13.099897540983607, 0.6693132795197584),
                    (9.690202702702702, 0.17513593019922968),
                    (10.06980802792321, 0.7947848617875114),
                    (14.605333333333334, 0.6012305898922827),
                    (36.30294396961064, 9.24123672148819),
                    (16.9130060292851, 0.7452868131028928),
                    (40.244078460399706, 3.7700993678269037),
                    (13.161603102779575, 0.1975347910041472),
                    (11.222161868549701, 0.7021223062972527)]

# time to download the model
DOWNLOAD_TIMES = [(112.42869743589742, 14.456734719659513),
                  (117.26870618556701, 12.549951446132013),
                  (13.059666666666667, 4.8700489616521185),
                  (47.50220338983051, 14.92128656898884),
                  (162.27864210526315, 32.562113378948396),
                  (99.46072058823529, 13.808785580783224),
                  (33.6347090909091, 25.00299299660141),
                  (216.25489393939392, 19.176465340447848),
                  (217.4117230769231, 20.757673955585453),
                  (98.38857297297298, 13.205048376808929),
                  (88.87509473684209, 23.152936862511545),
                  (66.96994262295081, 16.682497150763503),
                  (36.668852040816326, 13.759109844677598),
                  (149.31716326530614, 26.018185409516104),
                  (139.847, 80.04755583050091),
                  (54.97624444444445, 16.645170929316794)]

# time to upload the model
UPLOAD_TIMES = [(192.28497409326425, 21.537450985376967),
                (194.60103626943004, 24.194406902237056),
                (20.0, 0.0),
                (52.43859649122807, 5.047207127169352),
                (182.82417582417582, 14.793519078918195),
                (143.38059701492537, 7.910690646792151),
                (30.695652173913043, 9.668122350904568),
                (430.95360824742266, 54.97790476867727),
                (348.3174603174603, 30.14347985347738),
                (141.43715846994536, 5.271340868190727),
                (158.7433155080214, 64.87526819391198),
                (81.06086956521739, 7.003461202082419),
                (32.60621761658031, 5.0418315093016615),
                (281.5388601036269, 90.60338778706557),
                (194.34065934065933, 36.6519776778435),
                (66.53787878787878, 16.456280602190606)]

logger = getLogger(__name__)
# This catches PyTorch UserWarnings for CPU
warnings.filterwarnings("ignore", category=UserWarning)

CollaboratorTimeStats = namedtuple('CollaboratorTimeStats',
                                    [
                                        'validation_mean',
                                        'training_mean',
                                        'download_speed_mean',
                                        'upload_speed_mean',
                                        'validation_std',
                                        'training_std',
                                        'download_speed_std',
                                        'upload_speed_std',
                                    ]
                                    )

def gen_collaborator_time_stats(collaborator_names, seed=0xFEEDFACE):

    np.random.seed(seed)

    stats = {}    
    for col in collaborator_names:
        ml_index    = np.random.randint(len(VALIDATION_TIMES))
        validation  = VALIDATION_TIMES[ml_index]
        training    = TRAINING_TIMES[ml_index]
        net_index   = np.random.randint(len(DOWNLOAD_TIMES))
        download    = DOWNLOAD_TIMES[net_index]
        upload      = UPLOAD_TIMES[net_index]

        stats[col] = CollaboratorTimeStats(validation_mean=validation[0],
                                           training_mean=training[0],
                                           download_speed_mean=download[0],
                                           upload_speed_mean=upload[0],
                                           validation_std=validation[1],
                                           training_std=training[1],
                                           download_speed_std=download[1],
                                           upload_speed_std=upload[1])
    return stats

def compute_times_per_collaborator(collaborator_names,
                                   training_collaborators,
                                   epochs_per_round,
                                   collaborator_data,
                                   collaborator_time_stats,
                                   round_num):
    np.random.seed(round_num)
    times = {}
    for col in collaborator_names:
        time = 0

        # stats
        stats = collaborator_time_stats[col]

        # download time
        download_time = np.random.normal(loc=stats.download_speed_mean,
                                         scale=stats.download_speed_std)
        download_time = max(1, download_time)
        time += download_time

        # data loader
        data = collaborator_data[col]

        # validation time
        data_size = data.get_valid_data_size()
        validation_time_per = np.random.normal(loc=stats.validation_mean,
                                               scale=stats.validation_std)
        validation_time_per = max(1, validation_time_per)
        time += data_size * validation_time_per

        # only if training
        if col in training_collaborators:
            # training time
            data_size = data.get_train_data_size()
            training_time_per = np.random.normal(loc=stats.training_mean,
                                                 scale=stats.training_std)
            training_time_per = max(1, training_time_per)

            # training data size depends on the hparams
            data_size *= epochs_per_round
            time += data_size * training_time_per
            
            # if training, we also validate the locally updated model 
            data_size = data.get_valid_data_size()
            validation_time_per = np.random.normal(loc=stats.validation_mean,
                                                   scale=stats.validation_std)
            validation_time_per = max(1, validation_time_per)
            time += data_size * validation_time_per

            # upload time
            upload_time = np.random.normal(loc=stats.upload_speed_mean,
                                           scale=stats.upload_speed_std)
            upload_time = max(1, upload_time)
            time += upload_time
        
        times[col] = time
    return times


def get_metric(metric, fl_round, tensor_db):
    metric_name = metric
    target_tags = ('metric', 'validate_agg')
    return float(tensor_db.tensor_db.query("tensor_name == @metric_name and round == @fl_round and tags == @target_tags").nparray)

def collaborator_private_attributes(
        index, n_collaborators, train_csv, valid_csv, gandlf_config, device
    ):
        return {
            "train_csv": train_csv,
            "val_csv": valid_csv,
            "gandlf_config": gandlf_config,
            "device": device
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

    #fx.init('fets_challenge_workspace')
    
    from sys import path, exit

    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()

    gandlf_config_path = os.path.join(root, 'gandlf_config.yaml')

    print(f"TESTING ->>>>>>  Gandlf Config Path: {gandlf_config_path}")

    print(f"TESTING ->>>>>>  Work directory: {work}")

    path.append(str(root))
    path.insert(0, str(work))
    
    # create gandlf_csv and get collaborator names
    gandlf_csv_path = os.path.join(work, 'gandlf_paths.csv')
    # split_csv_path = os.path.join(work, institution_split_csv_filename)
    collaborator_names = construct_fedsim_csv(brats_training_data_parent_dir,
                                              institution_split_csv_filename,
                                              0.8,
                                              gandlf_csv_path)
    
    print(f'TESTING ->>>>>> Collaborator names: {collaborator_names}')

    aggregation_wrapper = CustomAggregationWrapper(aggregation_function) # ---> [TODO] Set the aggregation function in the workflow

    overrides = {
        'aggregator.settings.rounds_to_train': rounds_to_train,
        'aggregator.settings.db_store_rounds': db_store_rounds,
        'tasks.train.aggregation_type': aggregation_wrapper,
        'task_runner.settings.device': device,
    }


    # Update the plan if necessary
    # # [Kush - Flow] -> Update the Plan with the overrides
    # ---> Not required in workflow
    #plan = fx.update_plan(overrides)

    # if not include_validation_with_hausdorff:
    #     plan.config['task_runner']['settings']['fets_config_dict']['metrics'] = ['dice','dice_per_label']

    # # Overwrite collaborator names
    # plan.authorized_cols = collaborator_names
    # # overwrite datapath values with the collaborator name itself
    # for col in collaborator_names:
    #     # [Kush - Flow] -> Collaborator data path dictionary
    #     plan.cols_data_paths[col] = col

    # get the data loaders for each collaborator
    # [Kush - Flow] -> def get_data_loader(self, collaborator_name): Builds the DataLoader for the collaborator based on plan
    # --> Not required for workflow
    # collaborator_data_loaders = {col: copy(plan).get_data_loader(col) for col in collaborator_names}

    transformed_csv_dict = extract_csv_partitions(os.path.join(work, 'gandlf_paths.csv'))
    # get the task runner, passing the first data loader
    print('TESTING ->>>>>> Fetching TaskRunner ...')
    # for col in collaborator_data_loaders:
    #     #Insert logic to serialize train / val CSVs here
    #     # transformed_csv_dict[col]['train'].to_csv(os.path.join(work, 'seg_test_train.csv'))
    #     # transformed_csv_dict[col]['val'].to_csv(os.path.join(work, 'seg_test_val.csv'))
    #     transformed_csv_dict[col]['train'].to_csv(os.path.join(work, 'train.csv'))
    #     transformed_csv_dict[col]['val'].to_csv(os.path.join(work, 'valid.csv'))
    #     # [Kush - Flow] -> def get_task_runner(self, data_loader): Builds the TaskRunner and returns returns the taskrunner instance for the collaborator based on plan
    #     # ---> [[TODO]] Create coll priv_attributes as per csv dictionary.
    #     task_runner = copy(plan).get_task_runner(collaborator_data_loaders[col])

    aggregator = Aggregator()

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
                num_cpus=0.0,
                num_gpus=0.0,
                # arguments required to pass to callable
                index=idx,
                n_collaborators=len(collaborator_names),
                train_csv=train_csv_path,
                valid_csv=val_csv_path,
                gandlf_config=gandlf_config_path,
                device=device
            )
        )

    local_runtime = LocalRuntime(
        aggregator=aggregator, collaborators=collaborators, backend="single_process"
    )

    logger.info(f"Local runtime collaborators = {local_runtime.collaborators}")

    model = FeTSChallengeModel(gandlf_config_path)
    top_model_accuracy = 0
    # optimizers = {
    #     collaborator.name: default_optimizer(model, optimizer_type=args.optimizer_type)
    #     for collaborator in collaborators
    # }
    # flflow = FederatedFlow(
    #     model,
    #     optimizers,
    #     device,
    #     args.comm_round,
    #     top_model_accuracy,
    #     args.flow_internal_loop_test,
    # )

    flflow = FeTSFederatedFlow(
        model,
        1
    )

    flflow.runtime = local_runtime
    flflow.run()

    # [Kush - Flow] -> Commenting as pretrained model is not used.
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

    # [Kush - Flow] -> [TODO] Compression Pipeline
    # tensor_pipe = plan.get_tensor_pipe()

    # # Initialize model weights
    # # [Kush - FLow] - [TODO] How to set the initial state in the workflow
    # init_state_path = plan.config['aggregator']['settings']['init_state_path']
    # tensor_dict, _ = split_tensor_dict_for_holdouts(logger, task_runner.get_tensor_dict(False))

    # model_snap = utils.construct_model_proto(tensor_dict=tensor_dict,
    #                                          round_number=0,
    #                                          tensor_pipe=tensor_pipe)

    # utils.dump_proto(model_proto=model_snap, fpath=init_state_path)

    # # [Kush - Flow] ->Fetch the required aggregator from plan
    # # --> [SKIP] Not required for workflow as we will we creating aggregator and setting to runtime
    # # get the aggregator, now that we have the initial weights file set up
    # logger.info('Creating aggregator...')
    # aggregator = plan.get_aggregator()
    # # manually override the aggregator UUID (for checkpoint resume when rounds change)
    # aggregator.uuid = 'aggregator'
    # aggregator._load_initial_tensors()

    # # create our collaborators
    # # [Kush - Flow] ->Fetch the required COLLABORTAOR from plan
    # # --> [SKIP] Not required for workflow as we will we creating COLLABORATORS and setting to runtime
    # logger.info('Creating collaborators...')
    # collaborators = {col: copy(plan).get_collaborator(col, task_runner=task_runner, client=aggregator) for col in collaborator_names}

    # collaborator_time_stats = gen_collaborator_time_stats(plan.authorized_cols)

    # collaborators_chosen_each_round = {}
    # collaborator_times_per_round = {}

    # logger.info('Starting experiment')

    # total_simulated_time = 0
    # best_dice = -1.0
    # best_dice_over_time_auc = 0

    # # results dataframe data
    # experiment_results = {
    #         'round':[],
    #         'time': [],
    #         'convergence_score': [],
    #         'round_dice': [],
    #         'dice_label_0': [],
    #         'dice_label_1': [],
    #         'dice_label_2': [],
    #         'dice_label_4': [],
    #     }
    # if include_validation_with_hausdorff:
    #     experiment_results.update({
    #         'hausdorff95_label_0': [],
    #         'hausdorff95_label_1': [],
    #         'hausdorff95_label_2': [],
    #         'hausdorff95_label_4': [],
    #         })
        
    # # [Kush-Flow] [TODO] Will check later
    # # if restore_from_checkpoint_folder is None:
    # #     checkpoint_folder = setup_checkpoint_folder()
    # #     logger.info(f'\nCreated experiment folder {checkpoint_folder}...')
    # #     starting_round_num = 0
    # # else:
    # #     if not Path(f'checkpoint/{restore_from_checkpoint_folder}').exists():
    # #         logger.warning(f'Could not find provided checkpoint folder: {restore_from_checkpoint_folder}. Exiting...')
    # #         exit(1)
    # #     else:
    # #         logger.info(f'Attempting to load last completed round from {restore_from_checkpoint_folder}')
    # #         state = load_checkpoint(restore_from_checkpoint_folder)
    # #         checkpoint_folder = restore_from_checkpoint_folder

    # #         [loaded_collaborator_names, starting_round_num, collaborator_time_stats, 
    # #          total_simulated_time, best_dice, best_dice_over_time_auc, 
    # #          collaborators_chosen_each_round, collaborator_times_per_round, 
    # #          experiment_results, summary, agg_tensor_db] = state

    # #         if loaded_collaborator_names != collaborator_names:
    # #             logger.error(f'Collaborator names found in checkpoint ({loaded_collaborator_names}) '
    # #                          f'do not match provided collaborators ({collaborator_names})')
    # #             exit(1)

    # #         logger.info(f'Previous summary for round {starting_round_num}')
    # #         logger.info(summary)

    # #         starting_round_num += 1
    # #         aggregator.tensor_db.tensor_db = agg_tensor_db
    # #         aggregator.round_number = starting_round_num


    # for round_num in range(starting_round_num, rounds_to_train):
    #     # pick collaborators to train for the round
    #     # [Kush - Flow] -> Choose Training Collaborators
    #     # ---> [TODO] In flow based API's, in start we can pass as foreach = 'collaborators' 
    #     training_collaborators = choose_training_collaborators(collaborator_names,
    #                                                            aggregator.tensor_db._iterate(),
    #                                                            round_num,
    #                                                            collaborators_chosen_each_round,
    #                                                            collaborator_times_per_round)
        
    #     logger.info('Collaborators chosen to train for round {}:\n\t{}'.format(round_num, training_collaborators))

    #     # save the collaborators chosen this round
    #     collaborators_chosen_each_round[round_num] = training_collaborators

    #     # get the hyper-parameters from the competitor
    #     # [KUSH - Flow] --> Need to set how to set hyper parameters in the workflow
    #     # --> [TODO] Set some private attribute for the collaborator
    #     hparams = training_hyper_parameters_for_round(collaborator_names,
    #                                                   aggregator.tensor_db._iterate(),
    #                                                   round_num,
    #                                                   collaborators_chosen_each_round,
    #                                                   collaborator_times_per_round)

    #     learning_rate, epochs_per_round = hparams

    #     if (epochs_per_round is None):
    #         logger.warning('Hyper-parameter function warning: function returned None for "epochs_per_round". Setting "epochs_per_round" to 1')
    #         epochs_per_round = 1
        
    #     hparam_message = "\n\tlearning rate: {}".format(learning_rate)

    #     hparam_message += "\n\tepochs_per_round: {}".format(epochs_per_round)

    #     logger.info("Hyper-parameters for round {}:{}".format(round_num, hparam_message))

    #     # cache each tensor in the aggregator tensor_db
    #     hparam_dict = {}
    #     tk = TensorKey(tensor_name='learning_rate',
    #                     origin=aggregator.uuid,
    #                     round_number=round_num,
    #                     report=False,
    #                     tags=('hparam', 'model'))
    #     hparam_dict[tk] = np.array(learning_rate)
    #     tk = TensorKey(tensor_name='epochs_per_round',
    #                     origin=aggregator.uuid,
    #                     round_number=round_num,
    #                     report=False,
    #                     tags=('hparam', 'model'))
    #     hparam_dict[tk] = np.array(epochs_per_round)
    #     # [Kush - FLow] -> [TODO] How to cache the tensor in the workflow ?
    #     aggregator.tensor_db.cache_tensor(hparam_dict)

    #     # pre-compute the times for each collaborator
    #     # [Kush - Flow] [TODO] What is the use of this ?
    #     times_per_collaborator = compute_times_per_collaborator(collaborator_names,
    #                                                             training_collaborators,
    #                                                             epochs_per_round,
    #                                                             collaborator_data_loaders,
    #                                                             collaborator_time_stats,
    #                                                             round_num)
    #     collaborator_times_per_round[round_num] = times_per_collaborator

    #     # [Kush - Flow] -> Not required in workflow
    #     aggregator.assigner.set_training_collaborators(training_collaborators)

    #     # update the state in the aggregation wrapper
    #     # [Kush - Flow] -> [TODO] See how to pass this in the workflow as aggregation function and use in JOIN step
    #     aggregation_wrapper.set_state_data_for_round(collaborators_chosen_each_round, collaborator_times_per_round)

    #     # turn the times list into a list of tuples and sort it
    #     times_list = [(t, col) for col, t in times_per_collaborator.items()]
    #     times_list = sorted(times_list)

    #     # now call each collaborator in order of time
    #     # FIXME: this doesn't break up each task. We need this if we're doing straggler handling
    #     # [Kush - Flow] -> Below codeblock is not required in workflow as below two lines will be handled by the workflow
    #     # ---> [TODO] Create LocalRunTime using ray bakcend and do flow.run() to start the training
    #     for t, col in times_list:
    #         # set the task_runner data loader
    #         task_runner.data_loader = collaborator_data_loaders[col]

    #         # run the collaborator
    #         collaborators[col].run_simulation()
            
    #         logger.info("Collaborator {} took simulated time: {} minutes".format(col, round(t / 60, 2)))

    #     # the round time is the max of the times_list
    #     round_time = max([t for t, _ in times_list])
    #     total_simulated_time += round_time

    #    # [Kush - Flow] --> [TODO] How to set these metrics in the workflow and save the checkpoint ??
    #     # get the performace validation scores for the round
    #     round_dice = get_metric('valid_dice', round_num, aggregator.tensor_db)
    #     dice_label_0 = get_metric('valid_dice_per_label_0', round_num, aggregator.tensor_db)
    #     dice_label_1 = get_metric('valid_dice_per_label_1', round_num, aggregator.tensor_db)
    #     dice_label_2 = get_metric('valid_dice_per_label_2', round_num, aggregator.tensor_db)
    #     dice_label_4 = get_metric('valid_dice_per_label_4', round_num, aggregator.tensor_db)
    #     if include_validation_with_hausdorff:
    #         hausdorff95_label_0 = get_metric('valid_hd95_per_label_0', round_num, aggregator.tensor_db)
    #         hausdorff95_label_1 = get_metric('valid_hd95_per_label_1', round_num, aggregator.tensor_db)
    #         hausdorff95_label_2 = get_metric('valid_hd95_per_label_2', round_num, aggregator.tensor_db)
    #         hausdorff95_label_4 = get_metric('valid_hd95_per_label_4', round_num, aggregator.tensor_db)

    #     # update best score
    #     if best_dice < round_dice:
    #         best_dice = round_dice
    #         # Set the weights for the final model
    #         if round_num == 0:
    #             # here the initial model was validated (temp model does not exist)
    #             logger.info(f'Skipping best model saving to disk as it is a random initialization.')
    #         elif not os.path.exists(f'checkpoint/{checkpoint_folder}/temp_model.pkl'):
    #             raise ValueError(f'Expected temporary model at: checkpoint/{checkpoint_folder}/temp_model.pkl to exist but it was not found.')
    #         else:
    #             # here the temp model was the one validated
    #             shutil.copyfile(src=f'checkpoint/{checkpoint_folder}/temp_model.pkl',dst=f'checkpoint/{checkpoint_folder}/best_model.pkl')
    #             logger.info(f'Saved model with best average binary DICE: {best_dice} to ~/.local/workspace/checkpoint/{checkpoint_folder}/best_model.pkl')

    #     ## RUN VALIDATION ON INTERMEDIATE CONSENSUS MODEL
    #     # set the task_runner data loader
    #     # task_runner.data_loader = collaborator_data_loaders[col]

    #     ## CONVERGENCE METRIC COMPUTATION
    #     # update the auc score
    #     best_dice_over_time_auc += best_dice * round_time

    #     # project the auc score as remaining time * best dice
    #     # this projection assumes that the current best score is carried forward for the entire week
    #     projected_auc = (MAX_SIMULATION_TIME - total_simulated_time) * best_dice + best_dice_over_time_auc
    #     projected_auc /= MAX_SIMULATION_TIME

    #     # End of round summary
    #     summary = '"**** END OF ROUND {} SUMMARY *****"'.format(round_num)
    #     summary += "\n\tSimulation Time: {} minutes".format(round(total_simulated_time / 60, 2))
    #     summary += "\n\t(Projected) Convergence Score: {}".format(projected_auc)
    #     summary += "\n\tDICE Label 0: {}".format(dice_label_0)
    #     summary += "\n\tDICE Label 1: {}".format(dice_label_1)
    #     summary += "\n\tDICE Label 2: {}".format(dice_label_2)
    #     summary += "\n\tDICE Label 4: {}".format(dice_label_4)
    #     if include_validation_with_hausdorff:
    #         summary += "\n\tHausdorff95 Label 0: {}".format(hausdorff95_label_0)
    #         summary += "\n\tHausdorff95 Label 1: {}".format(hausdorff95_label_1)
    #         summary += "\n\tHausdorff95 Label 2: {}".format(hausdorff95_label_2)
    #         summary += "\n\tHausdorff95 Label 4: {}".format(hausdorff95_label_4)


    #     experiment_results['round'].append(round_num)
    #     experiment_results['time'].append(total_simulated_time)
    #     experiment_results['convergence_score'].append(projected_auc)
    #     experiment_results['round_dice'].append(round_dice)
    #     experiment_results['dice_label_0'].append(dice_label_0)
    #     experiment_results['dice_label_1'].append(dice_label_1)
    #     experiment_results['dice_label_2'].append(dice_label_2)
    #     experiment_results['dice_label_4'].append(dice_label_4)
    #     if include_validation_with_hausdorff:
    #         experiment_results['hausdorff95_label_0'].append(hausdorff95_label_0)
    #         experiment_results['hausdorff95_label_1'].append(hausdorff95_label_1)
    #         experiment_results['hausdorff95_label_2'].append(hausdorff95_label_2)
    #         experiment_results['hausdorff95_label_4'].append(hausdorff95_label_4)
    #     logger.info(summary)

    #     if save_checkpoints:
    #         logger.info(f'Saving checkpoint for round {round_num}')
    #         logger.info(f'To resume from this checkpoint, set the restore_from_checkpoint_folder parameter to \'{checkpoint_folder}\'')
    #         save_checkpoint(checkpoint_folder, aggregator, 
    #                         collaborator_names, collaborators,
    #                         round_num, collaborator_time_stats, 
    #                         total_simulated_time, best_dice, 
    #                         best_dice_over_time_auc, 
    #                         collaborators_chosen_each_round, 
    #                         collaborator_times_per_round,
    #                         experiment_results,
    #                         summary)

    #     # if the total_simulated_time has exceeded the maximum time, we break
    #     # in practice, this means that the previous round's model is the last model scored,
    #     # so a long final round should not actually benefit the competitor, since that final
    #     # model is never globally validated
    #     if total_simulated_time > MAX_SIMULATION_TIME:
    #         logger.info("Simulation time exceeded. Ending Experiment")
    #         break

    #     # save the most recent aggregated model in native format to be copied over as best when appropriate
    #     # (note this model has not been validated by the collaborators yet)
    #     task_runner.rebuild_model(round_num, aggregator.last_tensor_dict, validation=True)
    #     task_runner.save_native(f'checkpoint/{checkpoint_folder}/temp_model.pkl')
        
            

    return pd.DataFrame.from_dict(experiment_results), checkpoint_folder


# High Level Things Required
# 1. Create Aggregator and see is private attributes setting required.
# 2. Create Collaborators and pass training/valid csv files as private attributes.(See what else can be passed as private attributes) - Done
# 3. How to set the hyperparameters for each round.
# 4. How to set the aggregation function.
# 5. Create FederatedFlow and what all steps are required. Define the functions for the steps. 
# 6. Create Federated Model Class 
# 7. Check how to update metrics after each round.
# 8. Check the requirement for setting times per collaborator. 
# 9. How to store checkpoint and restore from checkpoint.
# 10. How to set the initial state of the model.
# 
import pandas as pd
import pickle
from pathlib import Path
from glob import glob
from sys import exit
from logging import getLogger

logger = getLogger(__name__)

def setup_checkpoint_folder():
    # Create checkpoint
    Path("checkpoint").mkdir(parents=True, exist_ok=True)
    existing_checkpoints = glob('checkpoint/*')
    if len(existing_checkpoints) == 0:
      checkpoint_num = 1
    else:
      # Increment the existing checkpoint by 1
      checkpoint_num = sorted([int(x.replace('checkpoint/experiment_','')) for x in existing_checkpoints])[-1] + 1
    experiment_folder = f'experiment_{checkpoint_num}'
    checkpoint_folder = f'checkpoint/{experiment_folder}'
    Path(checkpoint_folder).mkdir(parents=True, exist_ok=False)
    return experiment_folder

def save_checkpoint(checkpoint_folder, aggregator,
                    collaborator_names, collaborators,
                    round_num, collaborator_time_stats, 
                    total_simulated_time, best_dice, 
                    best_dice_over_time_auc, 
                    collaborators_chosen_each_round, 
                    collaborator_times_per_round,
                    experiment_results,
                    summary):
    """
    Save latest checkpoint
    """
    # Save aggregator tensor_db
    aggregator.tensor_db.tensor_db.to_pickle(f'checkpoint/{checkpoint_folder}/aggregator_tensor_db.pkl')
    for col in collaborator_names:
        collaborators[col].tensor_db.tensor_db.to_pickle(f'checkpoint/{checkpoint_folder}/{col}_tensor_db.pkl')
    with open(f'checkpoint/{checkpoint_folder}/state.pkl', 'wb') as f:
        pickle.dump([collaborator_names, round_num, collaborator_time_stats, total_simulated_time, 
                     best_dice, best_dice_over_time_auc, collaborators_chosen_each_round, 
                     collaborator_times_per_round, experiment_results, summary], f)

def load_checkpoint(checkpoint_folder):
    """
    Reload checkpoint from 'checkpoint/experiment_*'
    """
    aggregator_tensor_db = pd.read_pickle(f'checkpoint/{checkpoint_folder}/aggregator_tensor_db.pkl')
    with open(f'checkpoint/{checkpoint_folder}/state.pkl', 'rb') as f:
         state = pickle.load(f)

    # load each collaborator tensor_db
    collaborator_names = state[0]
    collaborator_tensor_dbs = {}
    for col in collaborator_names:
        collaborator_tensor_dbs[col] = pd.read_pickle(f'checkpoint/{checkpoint_folder}/{col}_tensor_db.pkl')
    
    return state + [aggregator_tensor_db] + [collaborator_tensor_dbs]

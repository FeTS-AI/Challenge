# Provided by the FeTS Initiative (www.fets.ai) as part of the FeTS Challenge 2021

# Contributing Authors (alphabetical):
# Brandon Edwards (Intel)
# Micah Sheller (Intel)

import os

import numpy as np
import pandas as pd

from fets.data.base_utils import get_appropriate_file_paths_from_subject_dir


# some hard-coded keys
# feature stack order determines order of feature stack modes
# (so must be consistent across datasets used on a given model)
# dependency here with mode naming convention used in get_appropriate_file_paths_from_subject_dir
feature_modes = ['T1', 'T2', 'FLAIR', 'T1CE']
label_tag = 'Label'

# using numerical header names
numeric_header_names = {'T1': 1, 
                        'T2': 2, 
                        'FLAIR': 3, 
                        'T1CE': 4, 
                        'Label': 5}

# used to map from numbers back to keys
numeric_header_name_to_key = {value: key for key, value in numeric_header_names.items()}

# column names for dataframe used to create csv

# 0 is for the subject name, 1-4 for modes and 5 for label (as above)
train_val_headers = [0, 1, 2, 3, 4, 5]


def train_val_split(subdirs, percent_train, shuffle=True):
    
    if percent_train <=0 or percent_train >=1:
        raise ValueError('Percent train must be strictly between 0 and 1.')
        
    if len(subdirs) == 0:
        raise ValueError('An empty list was provided to split.')
    
    if shuffle:
        np.random.shuffle(subdirs)
        
    cutpoint = int(len(subdirs) * percent_train)
    if cutpoint == 0 or cutpoint == len(subdirs):
        raise ValueError('The amount of data and percent train led to either empty train or val.')
    
    train_subdirs = subdirs[:cutpoint]
    val_subdirs = subdirs[cutpoint:]
    
    return train_subdirs, val_subdirs


def paths_dict_to_dataframe(paths_dict, train_val_headers, numeric_header_name_to_key):
    
    # intitialize columns
    columns = {header: [] for header in train_val_headers}
    columns['TrainOrVal'] = [] 
    columns['Partition_ID'] = []
    
    for inst_name, inst_paths_dict in paths_dict.items():
        for usage in ['train', 'val']:
            for key_to_fpath in inst_paths_dict[usage]:
                columns['Partition_ID'].append(inst_name)
                columns['TrainOrVal'].append(usage)
                for header in train_val_headers:
                    if header == 0:
                        # grabbing the the data subfolder name as the subject id
                        columns[header].append(key_to_fpath['Subject_ID'])
                    else:
                        columns[header].append(key_to_fpath[numeric_header_name_to_key[header]])
    
    return pd.DataFrame(columns, dtype=str)
    

def construct_fedsim_csv(pardir, 
                         split_subdirs_path, 
                         percent_train, 
                         federated_simulation_train_val_csv_path):
    
    # read in the csv defining the subdirs per institution
    split_subdirs = pd.read_csv(split_subdirs_path, dtype=str)
    
    if not set(['Partition_ID', 'Subject_ID']).issubset(set(split_subdirs.columns)):
        raise ValueError("The provided csv at {} must at minimum contain the columns 'Partition_ID' and 'Subject_ID', but the columns are: {}".format(split_subdirs_path, list(split_subdirs.columns)))
    
    # sanity check that all subdirs provided in the dataframe are unique
    if not split_subdirs['Subject_ID'].is_unique:
        raise ValueError("Repeated references to the same data subdir were found in the 'Subject_ID' column of {}".format(split_subdirs_path))
    
    train_val_specified = ('TrainOrVal' in split_subdirs.columns)
    if train_val_specified:
        print("Inferring train/val split using 'TrainOrVal' column of split_subdirs csv")
    else:
        print("No 'TrainOrVal' column found in split_subdirs csv, so performing automated split using percent_train of {}".format(percent_train))
    
    
    inst_names = list(split_subdirs['Partition_ID'].unique())
    
    paths_dict = {inst_name: {'train': [], 'val': []} for inst_name in inst_names}
    for inst_name in inst_names:
        
        if train_val_specified:
            train_subdirs = list(split_subdirs[(split_subdirs['Partition_ID']==inst_name) & (split_subdirs['TrainOrVal']=='train')]['Subject_ID'])
            val_subdirs = list(split_subdirs[(split_subdirs['Partition_ID']==inst_name) & (split_subdirs['TrainOrVal']=='val')]['Subject_ID'])
            if len(train_subdirs) == 0:
                raise ValueError("Train/val split specified in {} for insitution {} indicates an empty training set.".format(split_subdirs_path, inst_name))
            if len(val_subdirs) == 0:
                raise ValueError("Train/val split specified in {} for insitution {} indicates an empty val set.".format(split_subdirs_path, inst_name))
        else:
            subdirs = list(split_subdirs[split_subdirs['Partition_ID']==inst_name]['Subject_ID'])
            train_subdirs, val_subdirs = train_val_split(subdirs=subdirs, percent_train=percent_train)
        
        for subdir in train_subdirs:
            inner_dict = get_appropriate_file_paths_from_subject_dir(os.path.join(pardir, subdir), include_labels=True)
            inner_dict['Subject_ID'] = subdir
            paths_dict[inst_name]['train'].append(inner_dict)
            
        for subdir in val_subdirs:
            inner_dict = get_appropriate_file_paths_from_subject_dir(os.path.join(pardir, subdir), include_labels=True)
            inner_dict['Subject_ID'] = subdir
            paths_dict[inst_name]['val'].append(inner_dict)
        
    # now construct the dataframe and save it as a csv
    df =  paths_dict_to_dataframe(paths_dict=paths_dict, 
                                  train_val_headers=train_val_headers, 
                                  numeric_header_name_to_key=numeric_header_name_to_key)
    
    df.to_csv(federated_simulation_train_val_csv_path, index=False)
    return list(sorted(df.Partition_ID.unique()))

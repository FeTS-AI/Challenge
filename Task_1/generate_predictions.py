#!/usr/bin/env python
# coding: utf-8

# # FeTS Challenge
# 
# Contributing Authors (alphabetical order):
# - Brandon Edwards (Intel)
# - Patrick Foley (Intel)
# - Micah Sheller (Intel)

from fets_challenge import model_outputs_to_disc
from pathlib import Path
import os
from sys import path
from logging import getLogger
from fets_challenge.gandlf_csv_adapter import construct_fedsim_csv, extract_csv_partitions

device='cpu'
logger = getLogger(__name__)
# infer participant home folder
home = str(Path.home())

# you will need to specify the correct experiment folder and the parent directory for
# the data you want to run inference over
checkpoint_folder='experiment_1'
#data_path = </PATH/TO/CHALLENGE_VALIDATION_DATA>
data_path = '/home/brats/MICCAI_FeTS2022_ValidationData'

working_directory= os.path.join(home, '.local/workspace/')

try:
    os.chdir(working_directory)
    logger.info(f"Directory changed to : {os.getcwd()}")
except FileNotFoundError:
    logger.info("Error: Directory not found.")
except PermissionError:
    logger.info("Error: Permission denied")

if checkpoint_folder is not None:
    best_model_path = os.path.join(working_directory, 'checkpoint', checkpoint_folder, 'best_model.pkl')
else:
    exit("No checkpoint folder found. Please provide a valid checkpoint folder. Exiting the experiment without inferencing")

# If the experiment is only run for a single round, use the temp model instead
if not Path(best_model_path).exists():
   best_model_path = os.path.join(working_directory, 'checkpoint', checkpoint_folder, 'temp_model.pkl')

if not Path(best_model_path).exists():
    exit("No model found. Please provide a valid checkpoint folder. Exiting the experiment without inferencing")

outputs_path = os.path.join(working_directory, 'checkpoint', checkpoint_folder, 'model_outputs')

validation_csv_filename=os.path.join(home, '.local/workspace/', 'validation.csv')


# Using this best model, we can now produce NIfTI files for model outputs 
# using a provided data directory

model_outputs_to_disc(data_path=data_path, 
                      validation_csv=validation_csv_filename,
                      output_path=outputs_path, 
                      native_model_path=best_model_path,
                      outputtag='',
                      device=device)

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
# from fets_challenge.gandlf_csv_adapter import construct_fedsim_csv, extract_classification_csv_partitions

device='cpu'

# infer participant home folder
home = str(Path.home())

# you will need to specify the correct experiment folder and the parent directory for
# the data you want to run inference over
checkpoint_folder='experiment_109'
print(f"inference for {checkpoint_folder}")
#data_path = </PATH/TO/CHALLENGE_VALIDATION_DATA>
data_path = '/home/locolinux2/datasets/RSNA_ASNR_MICCAI_BraTS2021_TestingData'

# you can keep these the same if you wish
best_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'best_model.pkl')
outputs_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'model_outputs')

validation_csv_filename='final_test.csv'


# Using this best model, we can now produce NIfTI files for model outputs 
# using a provided data directory

model_outputs_to_disc(data_path=data_path, 
                      validation_csv=validation_csv_filename,
                      output_path=outputs_path, 
                      native_model_path=best_model_path,
                      problem_type='classification',
                      outputtag='',
                      device=device)

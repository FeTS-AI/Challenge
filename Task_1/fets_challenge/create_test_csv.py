import os
import csv

# This script generates a CSV file named 'final_test.csv' in the current directory.
# The CSV file has two columns: 'Partition_ID' and 'Subject_ID'.
# 'Partition_ID' is set to -1 for all rows.
# 'Subject_ID' is populated with the names of folders in the specified directory.
# This is intended to replicate the format of 'validation.csv' in the FeTS Challenge repository.
# The generated CSV file can be used to list all test samples in your data path.

# Specify the directory containing the test data folders
data_dir = "/home/locolinux2/datasets/RSNA_ASNR_MICCAI_BraTS2021_TestingData"

dir_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

output_csv =  '/home/locolinux2/.local/workspace/final_test.csv'

# Write the CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Partition_ID', 'Subject_ID'])
    
    for directory in dir_list:
        writer.writerow(['-1', directory])

print(f'CSV file "{output_csv}" has been created with the folder names.')

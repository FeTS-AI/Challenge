from argparse import ArgumentParser
import json
from filecmp import cmpfiles
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("dir1", help="First directory to compare")
parser.add_argument("dir2", help="Second directory to compare")

args = parser.parse_args()

dir1 = Path(args.dir1)
dir2 = Path(args.dir2)

file_list_dir1 = [x.name for x in dir1.iterdir()]

match, mismatch, errors = cmpfiles(dir1, dir2, common=file_list_dir1, shallow=False)
print("Files to compare:")
print(file_list_dir1)
print("=====")
if len(mismatch) > 0 or len(errors) > 0:
    print("There were some mismatches in the output folders. Please report this!")
    with open('comparison.json', 'w') as f:
        json.dump({'match': match, 'mismatch': mismatch, 'errors': errors}, f, indent=4, sort_keys=True)
else:
    print("The files in both folders match. Great :)")
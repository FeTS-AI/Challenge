"""
This script is meant to be executed for each container on the remote FeTS platforms,
from the FeTS-CLI (which does the metric calculations).
"""

import argparse
import csv
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from typing import List


def run_container(sif: Path, in_dir: Path, out_dir: Path, subject_list: List[str], timeout_case: float):
    # These are defined here because they're independent of the user input
    container_in_dir = Path("/data")
    container_out_dir = Path("/out_dir")

    # build singularity bind mount paths (to include only test case images without segmentation)
    # this will result in a very long bind path, unfortunately.
    bind_str = ""
    num_cases = 0
    for case in in_dir.iterdir():
        if case.is_dir() and case.name in subject_list:
            subject_id = case.name
            t1_path = case / f"{subject_id}_brain_t1.nii.gz"
            t1c_path = case / f"{subject_id}_brain_t1ce.nii.gz"
            t2_path = case / f"{subject_id}_brain_t2.nii.gz"
            fl_path = case / f"{subject_id}_brain_flair.nii.gz"

            t1_path_container = container_in_dir / subject_id / f"{subject_id}_brain_t1.nii.gz"
            t1c_path_container = container_in_dir / subject_id / f"{subject_id}_brain_t1ce.nii.gz"
            t2_path_container = container_in_dir / subject_id / f"{subject_id}_brain_t2.nii.gz"
            fl_path_container = container_in_dir / subject_id / f"{subject_id}_brain_flair.nii.gz"

            # check if files exist
            missing_files = []
            if not t1_path.exists():
                missing_files.append(t1_path.name)
            if not t1c_path.exists():
                if (case / f"{subject_id}_brain_t1gd.nii.gz").exists():
                    t1c_path = case / f"{subject_id}_brain_t1gd.nii.gz"   # container path stays the same
                else:
                    missing_files.append(t1c_path.name)
            if not t2_path.exists():
                missing_files.append(t2_path.name)
            if not fl_path.exists():
                if (case / f"{subject_id}_brain_fl.nii.gz").exists():
                    fl_path = case / f"{subject_id}_brain_fl.nii.gz"   # container path stays the same
                else:
                    missing_files.append(fl_path.name)

            if len(missing_files) == 0:
                bind_str += (
                    f"{t1_path}:{t1_path_container}:ro,"
                    f"{t1c_path}:{t1c_path_container}:ro,"
                    f"{t2_path}:{t2_path_container}:ro,"
                    f"{fl_path}:{fl_path_container}:ro,"
                )
                num_cases += 1
            else:
                logging.error(
                    f"For case {case.name}, some files were missing: {', '.join(missing_files)}. "
                    f"Skipping this case..."
                )
        
    assert "_seg.nii.gz" not in bind_str, "Container should not have access to segmentation files!"
    
    bind_str += f"{out_dir}:/{container_out_dir}:rw"
    logging.debug(f"The bind path string is in total {len(bind_str)} characters long.")
    os.environ["SINGULARITY_BINDPATH"] = bind_str

    try:
        start_time = time.monotonic()

        singularity_str = (
            f"singularity run -C --writable-tmpfs --net --network=none --nv"
            f" {sif} -i {container_in_dir} -o {container_out_dir}"
        )
        logging.info("Running container with the command:")    
        logging.info(singularity_str)
        subprocess.run(
            shlex.split(singularity_str),
            timeout=timeout_case * num_cases,
            check=True
        )
        end_time = time.monotonic()
    except subprocess.TimeoutExpired as e:
        logging.error(f"Timeout of {timeout_case * num_cases} reached (for {num_cases} cases)."
                      f" Aborting...")
        raise e
    except subprocess.CalledProcessError as e:
        logging.error(f"Running container failed:")
        raise e
        # I re-raise exceptions here, because they would indicate that something is wrong with the submission

    logging.info(f"Execution time of the container: {end_time - start_time:0.2f} s")
    return end_time - start_time


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Start inference with FeTS singularity images...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--container_dir", type=str,
        help="Path to the folder where .sif files are located. All containers will be run."
    )
    parser.add_argument(
        "-i", "--input_dir",required=True,type=str,
        help=("Input data lies here. Make sure it has the correct folder structure!"),
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str,
        help="Folder where the output/predictions will be written to"
    )
    parser.add_argument(
        "-s", "--split_file", required=True, type=str,
        help="CSV-file that contains the split that should be used for evaluation."
    )
    parser.add_argument(
        "-t", "--timeout", default=200, required=False, type=float,
        help="Time budget PER CASE in seconds. Evaluation will be stopped after the total timeout of timeout * n_cases."
    )

    args = parser.parse_args()

    # Parse subject list from split file
    included_subjects = []
    with open(args.split_file, newline='') as csvfile:
        split_reader = csv.reader(csvfile)
        for row in split_reader:
            included_subjects.append(str(row[0]))
    logging.info(f"Read the following subjects from the split file: {', '.join(included_subjects)}")

    # get all container paths
    container_list = [x for x in Path(args.container_dir).iterdir() if x.suffix == '.sif']
    assert len(container_list) > 0
    for sif_file in container_list:
        logging.info("=========================================")
        logging.info(f"Starting evaluation of {sif_file.name}...")
        curr_out_dir = Path(args.output_dir) / sif_file.stem
        curr_out_dir.mkdir(exist_ok=True)
        run_container(
            sif_file,
            in_dir=Path(args.input_dir),
            out_dir=curr_out_dir,
            subject_list=included_subjects,
            timeout_case=args.timeout
        )
        
        # delete excess files in output folder here (all but the segmentations)
        accepted_filenames = [f"{subj}_seg.nii.gz" for subj in included_subjects]
        logging.info("Cleaning up output directory")
        for out_path in curr_out_dir.iterdir():
            if out_path.is_dir():
                logging.warning(f"Deleting directory in output folder: {out_path.name}")
                shutil.rmtree(out_path)
            elif out_path.name not in accepted_filenames:
                logging.warning(f"Deleting file in output folder which does follow naming convention: {out_path.name}")
                out_path.unlink()

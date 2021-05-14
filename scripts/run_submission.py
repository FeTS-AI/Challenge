"""
This script is meant to be executed for each container on the remote FeTS platforms,
from the FeTS-CLI (which does the metric calculations).
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time

# TODO use logging instead of prints -> how to sensibly combine with FeTS-CLI?

if __name__ == "__main__":

    print("Testing FeTS singularity image...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sif_file", type=str,
        help="Name of the container file you want to test. Should have the format 'teamXYZ.sif'"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        type=str,
        help=(
            "Input data lies here. Make absolutely sure it has the correct folder structure!"
        ),
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str, help="Folder where the output/predictions will be written to"
    )
    parser.add_argument(
        "--timeout", default=200, required=False, type=int,
        help="Time budget PER CASE in seconds. Evaluation will be stopped after the total timeout of timeout * n_cases."
    )

    args = parser.parse_args()

    TIME_PER_CASE = args.timeout   # seconds
    sif_file = args.sif_file
    input_dir = Path(args.input_dir)
    bind_all = args.bind_segs   # TODO remove this option
    output_dir = Path(args.output_dir)

    container_indir = Path("/data")
    container_outdir = Path("/out_dir")

    num_cases = len(list(input_dir.iterdir()))

    # build singularity bind mount paths (to include only test case images without segmentation)
    # this will result in a very long bind path, unfortunately.
    bind_str = ""
    for case in input_dir.iterdir():
        if not case.is_dir():
            continue
        subject_id = case.name
        t1_path = case / f"{subject_id}_t1.nii.gz"
        t1c_path = case / f"{subject_id}_t1ce.nii.gz"
        t2_path = case / f"{subject_id}_t2.nii.gz"
        fl_path = case / f"{subject_id}_flair.nii.gz"
        bind_str += (
            f"{t1_path}:{container_indir.joinpath(*t1_path.parts[-2:])}:ro,"
            f"{t1c_path}:{container_indir.joinpath(*t1c_path.parts[-2:])}:ro,"
            f"{t2_path}:{container_indir.joinpath(*t2_path.parts[-2:])}:ro,"
            f"{fl_path}:{container_indir.joinpath(*fl_path.parts[-2:])}:ro,"
        )
    assert "_seg.nii.gz" not in bind_str, "Container should not have access to segmentation files!"
    
    bind_str += f"{output_dir}:/{container_outdir}:rw"
    print(f"The bind path string is in total {len(bind_str)} characters long.")
    os.environ["SINGULARITY_BINDPATH"] = bind_str

    print("\nRunning container...")

    ret = ""
    try:
        start_time = time.monotonic()

        singularity_str = (
            f"singularity run -C --writable-tmpfs --net --network=none --nv"
            f" {sif_file} -i {container_indir} -o {container_outdir}"
        )
        
        print(singularity_str)
        ret = subprocess.run(
            shlex.split(singularity_str),
            timeout=TIME_PER_CASE * num_cases,
            check=True
        )
        end_time = time.monotonic()
    except subprocess.TimeoutExpired:
        print(f"Timeout of {TIME_PER_CASE * num_cases} reached (for {num_cases} cases)."
              f" Aborting...")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Running container failed:")
        raise e
        # I re-raise exceptions here, because they would indicate that something is wrong with the submission
    print(f"Execution time of the container: {end_time - start_time:0.2f} s")

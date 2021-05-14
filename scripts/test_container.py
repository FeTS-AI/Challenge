import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

# TODO use logging instead of prints

# deprecated; participants will be provided a folder with appropriate structure
def is_fets_patient_folder(data_path: Path):
    # expected structure: see https://fets-ai.github.io/Front-End/process_data
    # in principle, each patient dir has to have these four files (with prefix patient-id)
    expected_files = [
        '_t1.nii.gz',
        '_t1ce.nii.gz',
        '_t2.nii.gz',
        '_flair.nii.gz',
    ]
    counter = 0
    check_ok = True
    if data_path.is_file():
        print(f"Expected a directory, but found file: {data_path}")
        return False
    for p in data_path.iterdir():
        if p.is_dir():
            check_ok = False
            print(f"This directory should not be here: {p.absolute()}")
        match = False
        for suffix in expected_files:
            if p.name == data_path.name + suffix:
                match = True
                counter += 1
        if not match:
            check_ok = False
            print(f"{p} does not fit into FeTS-naming scheme.")
    if counter != 4:
        check_ok = False
        print(f"Expected 4 niftis, but found {counter}.")

    return check_ok


def is_fets_prediction_folder(pred_path: Path, data_path: Path, algorithm_id: str):
    error_list = []

    subjects = list(data_path.iterdir())
    predictions = list(pred_path.iterdir())
    if len(subjects) != len(predictions):
        print(f"Number of patients and predictions does not match!")   # warning
    # TODO decide how strict we are going to be during evaluation: What happens if more than the required files exist in output dir? check here?
    for case in subjects:
        match_found = False
        for pred in predictions:
            if pred.is_file():
                # TODO check filename convention again
                if pred.name == f"{case.name}_{algorithm_id}_seg.nii.gz":
                    match_found = True
                    break
            else:
                print(f"Found misplaced dir: {pred}. This will not be considered for evaluation!")   # warning
        if not match_found:
            print(f"Could not find a prediction for case {case.name}!")   # error
            error_list.append(case.name)
    print(f"encountered {len(error_list)} errors. Please check logs")
    return error_list


if __name__ == "__main__":
    # TODO 
    raise RuntimeError("this code is outdated! Will be updated once the run_submission.py is fixed.")

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
        "-o", "--output_dir", required=False, type=str, help="Folder where the output/predictions will be written to"
    )
    parser.add_argument(
        "--timeout", default=200, required=False, type=int,
        help="Time budget PER CASE in seconds. Evaluation will be stopped after the total timeout of timeout * n_cases."
    )
    parser.add_argument(
        "--bind_segs", action="store_true", help="Testing option for binding the whole FeTS data folder (including groundtruth seg.)."
    )
    parser.add_argument(
        "--test_bindings", action="store_true", help="Testing option for printing the directory content of /data to a file."
    )

    args = parser.parse_args()

    TIME_PER_CASE = args.timeout   # seconds
    sif_file = args.sif_file
    input_dir = Path(args.input_dir)
    bind_all = args.bind_segs   # TODO remove this option
    test_ls = args.test_bindings   # TODO remove this option
    
    tmp_dir = None
    if args.output_dir is None:
        # Max: not sure where this is saved eventually. Looks like all results are cleared after the run in this case?
        tmp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(tmp_dir.name)
    else:
        output_dir = Path(args.output_dir)

    num_cases = len(list(input_dir.iterdir()))

    # build singularity bind mount paths (to include only test case images without segmentation)
    # this will result in a very long bind path, unfortunately.
    if not bind_all:   # default case
        bind_str = ""
        container_dir = Path("/data")
        for case in input_dir.iterdir():
            if not case.is_dir():
                continue
            subject_id = case.name
            t1_path = case / f"{subject_id}_t1.nii.gz"
            t1c_path = case / f"{subject_id}_t1ce.nii.gz"
            t2_path = case / f"{subject_id}_t2.nii.gz"
            fl_path = case / f"{subject_id}_flair.nii.gz"
            bind_str += (
                f"{t1_path}:{container_dir.joinpath(*t1_path.parts[-2:])}:ro,"
                f"{t1c_path}:{container_dir.joinpath(*t1c_path.parts[-2:])}:ro,"
                f"{t2_path}:{container_dir.joinpath(*t2_path.parts[-2:])}:ro,"
                f"{fl_path}:{container_dir.joinpath(*fl_path.parts[-2:])}:ro,"
            )
        bind_str += f"{output_dir}:/out_dir:rw"
    else:
        bind_str = f"{input_dir}:/data:ro,{output_dir}:/out_dir:rw"
    print(f"The bind path string is in total {len(bind_str)} characters long.")
    os.environ["SINGULARITY_BINDPATH"] = bind_str

    print("\nRunning container...")

    ret = ""
    try:
        start_time = time.monotonic()

        if not test_ls:   # default case
            singularity_str = (
                f"singularity run -C --writable-tmpfs --net --network=none --nv"
                f" {sif_file} -i /data -o /out_dir"
            )
        else:
            # this is for checking that the segmentations are hidden
            singularity_str = (
                f"singularity exec -C --writable-tmpfs --net --network=none --nv"
                f" {sif_file} ls -aR /data > /out_dir/file_list_container"
            )
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
    except subprocess.CalledProcessError:
        print(f"Running container failed:")
        print(ret)
        exit(1)
    # TODO maybe "except else" block to catch any other exceptions and try to continue evaluation?
    print(f"Execution time of the container: {end_time - start_time:0.2f} s")

    # check output
    if not is_fets_prediction_folder(output_dir, input_dir, algorithm_id=sif_file.stem):
        print("Output folder test not passed. Please check messages above. Exiting...")
        exit(1)

    print("\nEvaluating predictions...")
    # Metrics are calculated in FeTS-CLI
    # TODO but for testing, it would be good if the participants can do it here.

    if tmp_dir is not None:
        tmp_dir.cleanup()
    # TODO maybe clean-up? But maybe it's also better to just wipe the output folder after evaluation of one algorithm is done (in CLI)

    print("Done.")

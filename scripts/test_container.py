import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

# TODO use logging instead of prints

# Max: Why necessary? to import evaluation script?
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


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


def is_fets_prediction_folder(pred_path: Path, data_path: Path):
    check_ok = True

    patients = list(data_path.iterdir())
    predictions = list(pred_path.iterdir())
    if len(patients) != len(predictions):
        print(f"Number of patients and predictions does not match!")
        check_ok = False
    # TODO decide how strict we are going to be during evaluation: What happens if more than the required files exist in output dir? check here?
    for case in patients:
        if not case.is_dir():
            print(
                f"Found misplaced file: {case}. This will not be considered for evaluation!")
            check_ok = False
            continue
        for pred in predictions:
            if pred.is_file():
                check_ok = pred.name == f"{case.name}.nii.gz"
                # TODO update filename convention here
            else:
                print(
                    f"Found misplaced dir: {pred}. This will not be considered for evaluation!")
                check_ok = False
    return check_ok


if __name__ == "__main__":

    print("Testing FeTS singularity image...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sif_file", type=str,
        help="Name of the container file you want to test"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        type=str,
        help=(
            "Input data lies here. Watch out for correct folder structure!"
        ),
    )
    parser.add_argument(
        "-o", "--output_dir", required=False, type=str, help="Folder where the output/predictions will be written too"
    )

    args = parser.parse_args()

    TIME_PER_CASE = 200   # seconds

    sif_file = args.sif_file
    input_dir = Path(args.input_dir)
    tmp_dir = None
    if args.output_dir is None:
        # Max: not sure where this is saved eventually. Looks like all results are cleared after the run in this case?
        tmp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(tmp_dir.name)
    else:
        output_dir = Path(args.output_dir)

    num_cases = len(list(input_dir.iterdir()))
    # # check input
    # for subdir in input_dir.iterdir():
    #     if not is_fets_patient_folder(subdir):
    #         print("Input folder test not passed. Please check messages above. Exiting...")
    #         exit(1)

    # build singularity bind mount paths (to include only test cases without segmentation)
    # this will result in a very long bind path, but I don't see another option.
    hide_segmentations = True
    if hide_segmentations:
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
    os.environ["SINGULARITY_BINDPATH"] = bind_str

    print("\nRunning container...")

    ret = ""
    try:
        start_time = time.monotonic()
        singularity_str = (
            f"singularity run -C --writable-tmpfs --net --network=none --nv"
            f" {sif_file} -i /data -o /out_dir"
        )
        # # this is for checking that the segmentations are hidden
        # singularity_str = (
        #     f"singularity exec -C --writable-tmpfs --net --network=none --nv"
        #     f" {sif_file} ls -aR /data"
        # )
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
    print(f"Execution time of the container: {end_time - start_time:0.2f} s")

    # # check output
    # if not is_fets_prediction_folder(output_dir, input_dir):
    #     print("Output folder test not passed. Please check messages above. Exiting...")
    #     exit(1)

    print("\nEvaluating predictions...")

    # TODO get evaluation code from Spyros
    brain_score = 0
    print("Brain-dataset score:", brain_score)

    if tmp_dir is not None:
        tmp_dir.cleanup()

    print("Done.")

"""
This script can be executed by participants locally to test their container before uploading it for a functionality test
"""

import argparse
import logging
from pathlib import Path
import tempfile

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
        logging.warning(f"Number of patients and predictions does not match!")
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
                logging.warning(f"Found misplaced dir: {pred}. This will not be considered for evaluation!")
        if not match_found:
            logging.error(f"Could not find a prediction for case {case.name}!")
            error_list.append(case.name)
    logging.info(f"Encountered {len(error_list)} errors. Please check the logs.")
    return error_list


if __name__ == "__main__":
    # This import works only if this is executed as a script
    from run_submission import run_container

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
        "--log_file", default='test.log', required=False, type=str,
        help="Path where logs should be stored."
    )
    args = parser.parse_args()

    logging.basicConfig(handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
                        encoding='utf-8', level=logging.INFO)
    logging.info("Testing FeTS singularity image...")


    TIME_PER_CASE = args.timeout   # seconds
    sif_file = Path(args.sif_file)
    input_dir = Path(args.input_dir)
    
    tmp_dir = None
    if args.output_dir is None:
        # Max: not sure where this is saved eventually. Looks like all results are cleared after the run in this case?
        tmp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(tmp_dir.name)
    else:
        output_dir = Path(args.output_dir)

    # maybe add input dir check here

    # This runs the container in the same way it is done in the testing phase
    run_container(sif_file, input_dir, output_dir, TIME_PER_CASE)

    # check output
    if not is_fets_prediction_folder(output_dir, input_dir, algorithm_id=sif_file.stem):
        # TODO: not sure if the algorithm id will be part of the container name.
        logging.error("Output folder test not passed. Please error messages in the logs.")
        # exit(1)

    logging.info("Evaluating predictions...")
    # Metrics are calculated in FeTS-CLI
    # TODO but for testing, it would be good if the participants can do it here. Ask Sarthak if it can be included here somehow...

    if tmp_dir is not None:
        logging.info("Cleaning up temporary output folder...")
        tmp_dir.cleanup()

    logging.info("Done.")

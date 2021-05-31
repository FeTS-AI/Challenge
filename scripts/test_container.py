"""
This script can be executed by participants locally to test their container before uploading it for a functionality test
"""

import argparse
import logging
from pathlib import Path
import tempfile


def check_prediction_folder(pred_path: Path, data_path: Path):
    missing_cases = []

    subjects = list(data_path.iterdir())
    predictions = list(pred_path.iterdir())
    for case in subjects:
        match_found = -1
        for i, pred in enumerate(predictions):
            # anything in between the case identifier and the suffix will be ignored
            if pred.name.startswith(case.name) and pred.name.endswith("_seg.nii.gz"):
                match_found = i
                break
        if match_found >= 0:
            predictions.pop(match_found)
        else:
            missing_cases.append(case.name)
    if len(predictions) > 0:
        logging.error(f"The output folder contains files/folders that do not comply with the naming convention:\n{[str(el) for el in predictions]}")
        return False, missing_cases
    if len(missing_cases) > 0:
        logging.warning(f"{len(missing_cases)} cases did not have a prediction:\n{missing_cases}")
    return True, missing_cases


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
        "--input_dir", required=True, type=str,
        help="Input data lies here. Make sure it has the correct folder structure!",
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
    # TODO maybe add "compute_metrics"

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
    runtime = run_container(sif_file, input_dir, output_dir, TIME_PER_CASE)

    # check output
    folder_ok, missing_cases = check_prediction_folder(output_dir, input_dir)
    if len(missing_cases) > 0:
        # TODO depending on how the evaluation works, do something here
        pass
    if not folder_ok:
        logging.error("Output folder test not passed. Please check the error messages in the logs.")
        exit(1)

    logging.info("Evaluating predictions...")
    # Metrics are calculated in FeTS-CLI
    # TODO but for testing, it would be good if the participants can do it here. Ask Sarthak if it can be included here somehow...

    if tmp_dir is not None:
        logging.info("Cleaning up temporary output folder...")
        tmp_dir.cleanup()

    logging.info("Done.")

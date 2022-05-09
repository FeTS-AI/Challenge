from argparse import ArgumentParser
from pathlib import Path
import shutil


TOY_CASE_IDS = ["FeTS2022_01151", "FeTS2022_00805", "FeTS2022_00311"]


def main():
    parser = ArgumentParser(
        usage="This script helps you extracting the toy test cases used for sanity checks in the FeTS challenge. "
        "It assumes that you have downloaded the training data. "
        "Running it should leave you with a folder containing the test cases in the expected format."
    )
    parser.add_argument(
        "train_data_path",
        type=str,
        help="Path of the directory that contains your locally stored FeTS 2022 training data.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path of the directory where the extracted toy test cases should be stored.",
    )
    args = parser.parse_args()
    train_data_path = Path(args.train_data_path)
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)
    if train_data_path == output_path:
        raise ValueError(
            "Please specify a different folder for output to avoid overwriting training data."
        )

    print(f"Copying {TOY_CASE_IDS} from {train_data_path} to {output_path}...")
    for case_id in TOY_CASE_IDS:
        # copy files to output dir with different name (_brain_)
        output_case_dir = output_path / case_id
        output_case_dir.mkdir(exist_ok=True)
        for nifti in (train_data_path / case_id).iterdir():
            if nifti.name.endswith(".nii.gz"):
                suffix = nifti.name.split("_")[-1]
                shutil.copy2(nifti, output_case_dir / f"{case_id}_brain_{suffix}")
    print("Done.")


if __name__ == "__main__":
    main()

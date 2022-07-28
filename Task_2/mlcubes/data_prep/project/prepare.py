import csv
from pathlib import Path
import random
import shutil
from typing import List
from tqdm import tqdm


def copy_subject(subject_dir: Path, output_dir_data: Path, output_dir_labels: Path):
    subj_id = subject_dir.name
    # it's possible that minor naming differences are present. Accepted options for each modality are below.
    # input format:
    # <subject_id>[_brain]_t1.nii.gz etc
    # <subject_id>[_brain]_final_seg.nii.gz
    # output format:
    # <subject_id>_brain_t1.nii.gz etc
    # <subject_id>_final_seg.nii.gz
    files_to_copy = {
        "t1": [f"{subj_id}_brain_t1.nii.gz", f"{subj_id}_t1.nii.gz"],
        "t1ce": [f"{subj_id}_brain_t1ce.nii.gz", f"{subj_id}_t1ce.nii.gz"],
        "t2": [f"{subj_id}_brain_t2.nii.gz", f"{subj_id}_t2.nii.gz"],
        "flair": [f"{subj_id}_brain_flair.nii.gz", f"{subj_id}_flair.nii.gz"],
        "seg": [
            f"{subj_id}_final_seg.nii.gz",
            f"{subj_id}_brain_final_seg.nii.gz",
            f"{subj_id}_seg.nii.gz",
            f"{subj_id}_brain_seg.nii.gz",
        ],
    }
    for k, fname_options in files_to_copy.items():
        for filename in fname_options:
            file_path = subject_dir / filename
            output_dir = output_dir_data / subj_id
            if k == "seg":
                output_dir = output_dir_labels
            output_dir.mkdir(exist_ok=True)
            if file_path.exists():
                shutil.copy2(file_path, output_dir / files_to_copy[k][0])
                break


def get_validation_subjects(
    data_path: Path,
    split_path: Path,
    val_size: float = 0.2,
    min_val: int = 10,
    seed=108493,
) -> List[Path]:
    subject_list = []
    if split_path.exists():
        with open(split_path, newline="", encoding="utf-8") as csvfile:
            split_reader = csv.reader(csvfile)
            for row in split_reader:
                if str(row[0]) == "data_uid":
                    continue
                subject_dir = data_path / str(row[0])
                if not subject_dir.exists():
                    raise FileNotFoundError(
                        f"The data folder {subject_dir} does not exist, but a corresponding subject was found in the validation split file. "
                        f"Please contact the organizers!"
                    )
                subject_list.append(subject_dir)
    else:
        print(
            f"WARNING: The file with data split information is not present. "
            f"Performing automatic split with val_size={val_size}. "
            f"Please contact the organizers if this causes errors."
        )
        all_subjects = []
        for x in Path(data_path).iterdir():
            # just to be sure there are no other folders that don't contain the actual data:
            if x.is_dir() and len(list(x.glob("*.nii.gz"))) > 0:
                all_subjects.append(x)
        min_val = min(min_val, len(all_subjects))
        num_val = max(int(val_size * len(all_subjects)), min_val)
        random.seed(seed)
        subject_list = random.sample(all_subjects, k=num_val)
    print(
        "Got {} subjects from the validation split: {}".format(
            len(subject_list),
            ", ".join([x.name for x in subject_list])
        )
    )
    return subject_list


def run_preparation(
    input_dir: str, output_data_dir: str, output_label_dir: str
) -> None:
    output_data_path = Path(output_data_dir)
    output_labels_path = Path(output_label_dir)
    output_data_path.mkdir(parents=True, exist_ok=True)
    output_labels_path.mkdir(parents=True, exist_ok=True)

    val_split_path = Path(input_dir) / "split_info" / "fets_phase2_split_1" / "val.csv"
    subject_dir_list = get_validation_subjects(Path(input_dir), val_split_path)
    print(f"Preparing {len(subject_dir_list)} subjects...")
    for subject_dir in tqdm(subject_dir_list):
        copy_subject(subject_dir, output_data_path, output_labels_path)

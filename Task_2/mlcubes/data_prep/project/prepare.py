import csv
import json
from pathlib import Path
import random
import shutil
from typing import List


def copy_subject(
    subject_dir: Path,
    output_dir_data: Path,
    output_dir_labels: Path,
    subject_alias: str = None,
):
    subj_id = subject_dir.name
    if subject_alias is None:
        subject_alias = subj_id
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
    target_files = {
        "t1": f"{subject_alias}_brain_t1.nii.gz",
        "t1ce": f"{subject_alias}_brain_t1ce.nii.gz",
        "t2": f"{subject_alias}_brain_t2.nii.gz",
        "flair": f"{subject_alias}_brain_flair.nii.gz",
        "seg": f"{subject_alias}_final_seg.nii.gz",
    }
    for modality, fname_options in files_to_copy.items():
        for filename in fname_options:
            # search for naming that exists in subject_dir
            output_dir = output_dir_data / subject_alias
            if modality == "seg":
                output_dir = output_dir_labels
            output_dir.mkdir(exist_ok=True)

            src_file_path = subject_dir / filename
            dst_file_path = output_dir / target_files[modality]
            if src_file_path.exists():
                # if no match is found for any option, don't copy anything. The sanity check will make sure no files are missing.
                shutil.copy2(src_file_path, dst_file_path)
                break


def _get_validation_subjects_splitfile(
    data_path: Path, max_size: int, seed: int, val_split_file: str = None
) -> List[Path]:
    """Note: This may return a list of size > max_size if there are more cases in the val_split_file"""

    # expect relative path in val_split_file
    val_split_file: Path = data_path / val_split_file
    if not val_split_file.exists():
        print(f"WARNING: The split file {data_path / val_split_file} does not exist.")
        return _get_validation_subjects(
            data_path=data_path, max_size=max_size, seed=seed
        )

    split_file_subjects = []
    # load subjects from split file
    with open(val_split_file, newline="", encoding="utf-8") as csvfile:
        split_reader = csv.reader(csvfile)
        for row in split_reader:
            if str(row[0]) == "data_uid":
                continue
            subject_dir = data_path / str(row[0])
            if not subject_dir.exists():
                print(
                    f"WARNING: The data folder {subject_dir} does not exist, but a corresponding subject was found in the validation split file. "
                    f"This will probably cause an error in the sanity check."
                )
            split_file_subjects.append(subject_dir.absolute())

    # Also get subjects not in splitfile and add them up to max_size
    subjects_not_in_splitfile = []
    for x in Path(data_path).iterdir():
        # just to be sure there are no other folders that don't contain the actual data:
        if (
            x.is_dir()
            and len(list(x.glob("*.nii.gz"))) > 0
            and x.absolute() not in split_file_subjects
        ):
            subjects_not_in_splitfile.append(x)

    random.seed(seed)
    num_additional_samples = min(
        len(subjects_not_in_splitfile), max(0, max_size - len(split_file_subjects))
    )
    return split_file_subjects + random.sample(
        subjects_not_in_splitfile, k=num_additional_samples
    )


def _get_validation_subjects(data_path: Path, max_size: int, seed: int) -> List[Path]:
    all_subjects = []
    for x in Path(data_path).iterdir():
        # just to be sure there are no other folders that don't contain the actual data:
        if x.is_dir() and len(list(x.glob("*.nii.gz"))) > 0:
            all_subjects.append(x)

    if len(all_subjects) > max_size:
        random.seed(seed)
        subject_list = random.sample(all_subjects, k=max_size)
    else:
        subject_list = all_subjects
    return subject_list


def get_validation_subjects(
    data_path: Path, max_size: int, seed: int, val_split_file: str = None
) -> List[Path]:
    """This function returns a list of subjects that should be used for evaluation. If there is a split file, it tries to include them in the set.
    Arguments:
        data_path: root directory containing all subject directories
        max_size: maximum number of subjects to add to the validation set (to limit inference time); may be exceeded in the case that val_split_file has more cases
        seed: used for sampling when more subjects than max_size are available
        val_split_file: path to split file (if it exists) from FeTS initiative (relative to data_path)
    """
    if val_split_file:
        subject_list = _get_validation_subjects_splitfile(
            data_path=data_path,
            max_size=max_size,
            seed=seed,
            val_split_file=val_split_file,
        )
    else:
        subject_list = _get_validation_subjects(
            data_path=data_path, max_size=max_size, seed=seed
        )
    print(
        "These {} subjects are in the validation split:\n{}".format(
            len(subject_list), ", ".join([x.name for x in subject_list])
        )
    )
    return subject_list


def compute_subject_aliases(subject_list: List[Path]) -> List[str]:
    # Enumeration is the simplest option; could also use hash functions
    return [f"FeTS22_Patient{idx:04d}" for idx, _ in enumerate(subject_list)]


def run_preparation(
    input_dir: str,
    output_data_dir: str,
    output_label_dir: str,
    max_val_size: int = 200,
    seed: int = 108493,
    val_split_file: str = None,
    anonymize_subjects: bool = True,
) -> None:
    """This function selects subjects from input_dir (and possibly the val_split_file) for validation and copies those to a the output paths.
    max_val_size, seed and val_split_file are passed to get_validation_subjects.
    """
    output_data_path = Path(output_data_dir)
    output_labels_path = Path(output_label_dir)
    output_data_path.mkdir(parents=True, exist_ok=True)
    output_labels_path.mkdir(parents=True, exist_ok=True)

    selected_subject_dirs = get_validation_subjects(
        Path(input_dir), max_size=max_val_size, seed=seed, val_split_file=val_split_file
    )
    print(f"Preparing {len(selected_subject_dirs)} subjects...")
    if anonymize_subjects:
        alias_list = compute_subject_aliases(selected_subject_dirs)
    else:
        alias_list = [None] * len(selected_subject_dirs)
    alias_mapping = {}
    for subject_dir, subject_alias in zip(selected_subject_dirs, alias_list):
        if anonymize_subjects:
            alias_mapping[subject_alias] = subject_dir.name
        copy_subject(
            subject_dir,
            output_data_path,
            output_labels_path,
            subject_alias=subject_alias,
        )

    # Output is saved to the medperf log. In the future, we may want to improve this.
    if anonymize_subjects:
        print("This is the mapping from aliases to subject IDs:")
        print(alias_mapping)
    else:
        print("These subject IDs were used for evaluation:")
        print([x.name for x in selected_subject_dirs])

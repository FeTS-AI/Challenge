from pathlib import Path
from typing import List, Tuple

import SimpleITK as sitk
import numpy as np


def check_subject_validity(
    subject_dir: Path, labels_dir: Path
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """Checks if all files exist. Also checks size, spacing and label set of images and mask.
    """
    missing_files = []
    wrong_size = []
    wrong_spacing = []
    wrong_labels = []

    files_to_check = [
        subject_dir / f"{subject_dir.name}_brain_t1.nii.gz",
        subject_dir / f"{subject_dir.name}_brain_t1ce.nii.gz",
        subject_dir / f"{subject_dir.name}_brain_t2.nii.gz",
        subject_dir / f"{subject_dir.name}_brain_flair.nii.gz",
        labels_dir / f"{subject_dir.name}_final_seg.nii.gz",
    ]
    # check image properties
    EXPECTED_SIZE = np.array([240, 240, 155])
    EXPECTED_SPACING = np.array([1.0, 1.0, 1.0])
    EXPECTED_LABELS = {0, 1, 2, 4}
    for file_ in files_to_check:
        if not file_.exists():
            missing_files.append(str(file_))
            continue
        image = sitk.ReadImage(str(file_))
        size_array = np.array(image.GetSize())
        spacing_array = np.array(image.GetSpacing())

        if not (EXPECTED_SIZE == size_array).all():
            wrong_size.append(str(file_))
        if not (EXPECTED_SPACING == spacing_array).all():
            wrong_spacing.append(str(file_))
        if file_.name.endswith("seg.nii.gz"):
            arr = sitk.GetArrayFromImage(image)
            found_labels = np.unique(arr)
            if len(set(found_labels).difference(EXPECTED_LABELS)) > 0:
                wrong_labels.append(str(file_))
    return missing_files, wrong_size, wrong_spacing, wrong_labels


def run_sanity_check(data_path: str, labels_path: str):
    check_successful = True
    for curr_subject_dir in Path(data_path).iterdir():
        if curr_subject_dir.is_dir():
            (
                missing_files,
                wrong_size,
                wrong_spacing,
                wrong_labels,
            ) = check_subject_validity(curr_subject_dir, Path(labels_path))
            if len(missing_files) > 0:
                check_successful = False
                print(
                    f"ERROR Files missing for subject {curr_subject_dir.name}:\n{missing_files}"
                )
            if len(wrong_size) > 0:
                check_successful = False
                print(f"ERROR: Image size is not [240,240,155] for:\n{wrong_size}")
            if len(wrong_spacing) > 0:
                check_successful = False
                print(f"ERROR: Image resolution is not [1,1,1] for:\n{wrong_spacing}")
            if len(wrong_labels) > 0:
                check_successful = False
                print(
                    f"ERROR: There were unexpected label values (not in [0, 1, 2, 4]) for:\n{wrong_labels}"
                )
    assert (
        check_successful
    ), "The sanity check discovered error(s). Please check the log above for details."
    print("Finished. All good!")

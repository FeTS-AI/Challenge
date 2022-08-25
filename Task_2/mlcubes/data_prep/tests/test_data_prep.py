import csv
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import SimpleITK as sitk

# ugly, but well...
sys.path.insert(0, str(Path(__file__).parents[1]))
print(sys.path)
from project.prepare import run_preparation, copy_subject
from project.sanity_check import run_sanity_check


def setup_dummy_data_dir(
    root_path: Path,
    split_file=None,
    num_cases=10,
    split_fraction=0.2,
    make_real_images=False,
):
    # This sets up a dummy fets data directory. Structure:
    # root_path /
    #     Case_id_0/
    #         Case_id_0_t1.nii.gz
    #         Case_id_0_t1ce.nii.gz
    #         Case_id_0_t2.nii.gz
    #         Case_id_0_flair.nii.gz
    #         Case_id_0_seg.nii.gz
    #     Case_id_1/
    #         ...
    modalities = ["t1", "t1ce", "t2", "flair", "seg"]
    case_ids = [f"ToyPatient{i:03d}" for i in range(num_cases)]
    for case in case_ids:
        case_dir = root_path / case
        case_dir.mkdir()
        for m in modalities:
            img_path = case_dir / f"{case}_{m}.nii.gz"
            if make_real_images:
                nda = np.zeros((155, 240, 240))  # BraTS dimensions
                img = sitk.GetImageFromArray(nda)
                sitk.WriteImage(img, str(img_path.absolute()))
            else:
                img_path.touch()

    split_file_cases = None
    if split_file:
        split_file_cases = random.sample(
            case_ids, k=int(split_fraction * len(case_ids))
        )
        split_path = root_path / split_file
        split_path.parent.mkdir(parents=True)
        with open(split_path, "w", encoding="utf-8") as f:
            csvwriter = csv.writer(f, delimiter=",")
            csvwriter.writerow(["data_uid"])
            csvwriter.writerows([[x] for x in split_file_cases])
    return case_ids, split_file_cases
    # should return the list of cases and (if split_file) list of validation cases


@pytest.mark.parametrize(
    "total_num_cases,max_val_size", [(100, 10), (100, 100), (10, 100)]
)
def test_data_prep_splitfile(tmp_path: Path, total_num_cases: int, max_val_size: int):
    split_file = "split_info/fets_phase2_split_1/val.csv"  # relative to data dir
    tmp_data_dir = tmp_path / "data"
    tmp_output_dir = tmp_path / "output_data"
    tmp_output_label_dir = tmp_path / "output_labels"
    # setup
    tmp_data_dir.mkdir()
    tmp_output_dir.mkdir()
    tmp_output_label_dir.mkdir()
    all_cases, split_file_cases = setup_dummy_data_dir(
        tmp_data_dir, num_cases=total_num_cases, split_file=split_file
    )

    run_preparation(
        input_dir=tmp_data_dir,
        output_data_dir=tmp_output_dir,
        output_label_dir=tmp_output_label_dir,
        max_val_size=max_val_size,
        val_split_file=split_file,
        anonymize_subjects=False,
    )
    output_cases = [x.name for x in tmp_output_dir.iterdir()]

    # no duplicates
    assert len(set(output_cases)) == len(output_cases)
    if max_val_size > len(split_file_cases):
        assert len(output_cases) == min(max_val_size, len(all_cases))
        assert set(output_cases).issubset(set(all_cases))
        assert set(split_file_cases).issubset(set(output_cases))
    else:
        assert set(output_cases) == set(split_file_cases)


@pytest.mark.parametrize("total_num_cases,max_val_size", [(100, 100)])
def test_data_prep_missing_splitfile(
    tmp_path: Path, total_num_cases: int, max_val_size: int
):
    split_file = "split_info/fets_phase2_split_1/val.csv"  # relative to data dir
    tmp_data_dir = tmp_path / "data"
    tmp_output_dir = tmp_path / "output_data"
    tmp_output_label_dir = tmp_path / "output_labels"
    # setup
    tmp_data_dir.mkdir()
    tmp_output_dir.mkdir()
    tmp_output_label_dir.mkdir()
    all_cases, split_file_cases = setup_dummy_data_dir(
        tmp_data_dir, num_cases=total_num_cases, split_file=split_file
    )
    (tmp_data_dir / split_file).unlink()  # delete to simulate missing split file

    run_preparation(
        input_dir=tmp_data_dir,
        output_data_dir=tmp_output_dir,
        output_label_dir=tmp_output_label_dir,
        max_val_size=max_val_size,
        val_split_file=split_file,
        anonymize_subjects=False,
    )
    # same as no splitfile
    output_cases = [x.name for x in tmp_output_dir.iterdir()]
    assert len(output_cases) == min(len(all_cases), max_val_size)
    assert set(output_cases).issubset(set(all_cases))


@pytest.mark.slow
@pytest.mark.parametrize("total_num_cases,max_val_size", [(10, 10),])
def test_data_prep_corrupted_splitfile(
    tmp_path: Path, total_num_cases: int, max_val_size: int
):
    split_file = "split_info/fets_phase2_split_1/val.csv"  # relative to data dir
    tmp_data_dir = tmp_path / "data"
    tmp_output_dir = tmp_path / "output_data"
    tmp_output_label_dir = tmp_path / "output_labels"
    # setup
    tmp_data_dir.mkdir()
    tmp_output_dir.mkdir()
    tmp_output_label_dir.mkdir()
    _, split_file_cases = setup_dummy_data_dir(
        tmp_data_dir,
        num_cases=total_num_cases,
        split_file=split_file,
        make_real_images=True,
    )

    # corrupt
    with open(tmp_data_dir / split_file, newline="", encoding="utf-8") as csvfile:
        split_reader = csv.reader(csvfile)
        lines = []
        for row in split_reader:
            lines.append(row)
        lines[-1][0] = "corrupted_entry"
        split_file_cases[-1] = "corrupted_entry"
    with open(
        tmp_data_dir / split_file, newline="", encoding="utf-8", mode="w"
    ) as csvfile:
        split_writer = csv.writer(csvfile)
        split_writer.writerows(lines)

    run_preparation(
        input_dir=tmp_data_dir,
        output_data_dir=tmp_output_dir,
        output_label_dir=tmp_output_label_dir,
        max_val_size=max_val_size,
        val_split_file=split_file,
    )
    # sanity check should fail in that case
    with pytest.raises(AssertionError):
        run_sanity_check(data_path=tmp_output_dir, labels_path=tmp_output_label_dir)


@pytest.mark.parametrize(
    "total_num_cases,max_val_size", [(100, 10), (100, 100), (10, 100)]
)
def test_data_prep_randomsplit(tmp_path: Path, total_num_cases: int, max_val_size: int):
    tmp_data_dir = tmp_path / "data"
    tmp_output_dir = tmp_path / "output_data"
    tmp_output_label_dir = tmp_path / "output_labels"
    # setup
    tmp_data_dir.mkdir()
    tmp_output_dir.mkdir()
    tmp_output_label_dir.mkdir()
    all_cases, _ = setup_dummy_data_dir(tmp_data_dir, num_cases=total_num_cases)

    run_preparation(
        input_dir=tmp_data_dir,
        output_data_dir=tmp_output_dir,
        output_label_dir=tmp_output_label_dir,
        max_val_size=max_val_size,
        anonymize_subjects=False,
    )
    output_cases = [x.name for x in tmp_output_dir.iterdir()]

    assert len(set(output_cases)) == len(output_cases)  # no duplicates
    assert len(output_cases) == min(len(all_cases), max_val_size)
    assert set(output_cases).issubset(set(all_cases))


@pytest.mark.parametrize(
    "include_brain,use_alias", [(False, True), (False, False), (True, True)]
)
def test_copy_subjects(tmp_path: Path, include_brain: bool, use_alias: bool):
    # need a directory with some FeTS files
    modalities = ["t1", "t1ce", "t2", "flair", "final_seg"]
    case_id = "Bruce Wayne"
    case_alias = "Batman"
    if not use_alias:
        case_alias = case_id
    subj_dir = tmp_path / case_id
    tmp_output_dir = tmp_path / "output_data"
    tmp_output_label_dir = tmp_path / "output_labels"
    subj_dir.mkdir()
    tmp_output_dir.mkdir()
    tmp_output_label_dir.mkdir()

    brain_placeholder = ""
    if include_brain:
        brain_placeholder = "_brain"
    for m in modalities:
        img_path = subj_dir / f"{case_id}{brain_placeholder}_{m}.nii.gz"
        img_path.touch()

    if use_alias:
        copy_subject(
            subj_dir, tmp_output_dir, tmp_output_label_dir, subject_alias=case_alias
        )
    else:
        copy_subject(subj_dir, tmp_output_dir, tmp_output_label_dir)

    expected_paths = set()
    for m in modalities:
        if m == "final_seg":
            expected_paths.add(tmp_output_label_dir / f"{case_alias}_{m}.nii.gz")
        else:
            expected_paths.add(
                tmp_output_dir / case_alias / f"{case_alias}_brain_{m}.nii.gz"
            )
    found_paths = set(tmp_output_dir.glob("**/*.nii.gz"))
    found_paths = found_paths.union(set(tmp_output_label_dir.glob("**/*.nii.gz")))

    assert expected_paths == found_paths


@pytest.mark.parametrize("total_num_cases,max_val_size", [(100, 10)])
def test_subject_anonymization(tmp_path: Path, total_num_cases: int, max_val_size: int):
    tmp_data_dir = tmp_path / "data"
    tmp_output_dir = tmp_path / "output_data"
    tmp_output_label_dir = tmp_path / "output_labels"
    # setup
    tmp_data_dir.mkdir()
    tmp_output_dir.mkdir()
    tmp_output_label_dir.mkdir()
    all_cases, _ = setup_dummy_data_dir(tmp_data_dir, num_cases=total_num_cases)

    run_preparation(
        input_dir=tmp_data_dir,
        output_data_dir=tmp_output_dir,
        output_label_dir=tmp_output_label_dir,
        max_val_size=max_val_size,
    )
    output_cases = [x.name for x in tmp_output_dir.iterdir()]

    assert len(set(output_cases)) == len(output_cases)  # no duplicates
    assert len(output_cases) == min(len(all_cases), max_val_size)
    assert set(output_cases).isdisjoint(set(all_cases))


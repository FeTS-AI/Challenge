"""Logic file"""
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.utils.utilities import helper


def pseudo_predict(subject_dir: Path, output_dir: Path):
    """
    In this dummy example, the four MR sequences are loaded from `subject_dir` and then class 0 is "predicted" from the t1-image,
    class 1 from t1ce etc., using a simple thresholding operation. The resulting segmentation is saved to `output_dir`.
    """
    # NOTE Please stick to this naming convention for your prediction!
    output_fname = output_dir / f"{subject_dir.name}.nii.gz"

    # NOTE FeTS structure: one folder for each test case (subject), containing four niftis.
    # Example: TODO check again with UPenn if this is up to date
    # Patient_001    # case identifier
    # │ Patient_001_brain_t1.nii.gz
    # │ Patient_001_brain_t1ce.nii.gz
    # │ Patient_001_brain_t2.nii.gz
    # │ Patient_001_brain_flair.nii.gz
    modalities = ["t1", "t1ce", "t2", "flair"]
    labels = [0, 1, 2, 4]
    seg_npy = None

    for mod, lab in zip(modalities, labels):
        img_path = next(subject_dir.glob(f"*_{mod}.nii.gz"))
        img_itk = sitk.ReadImage(str(img_path.absolute()))
        img_npy = sitk.GetArrayFromImage(img_itk)
        if seg_npy is None:
            seg_npy = np.zeros_like(img_npy)
        else:
            seg_npy[img_npy > np.percentile(img_npy, 95)] = lab

    # make sure segmentation occupies the same space
    seg_itk = sitk.GetImageFromArray(seg_npy)
    seg_itk.CopyInformation(img_itk)

    sitk.WriteImage(seg_itk, str(output_fname.absolute()))


def run_inference(
    input_folder: str,
    output_folder: str,
    checkpoint_folder: str,
    application_name: str,
    application_version: str,
) -> None:
    print(
        "*** code execution started:",
        application_name,
        "version:",
        application_version,
        "! ***",
    )
    in_folder = Path(input_folder)
    out_folder = Path(output_folder)
    params_folder = Path(checkpoint_folder)
    print("Number of subjects found in data path: ",
          len(list(in_folder.iterdir())))

    # no parameters are used in this example. This is just for illustration.
    if not params_folder.exists() or len(list(params_folder.iterdir())) == 0:
        raise FileNotFoundError(
            f"No model parameters found at {params_folder}")
    else:
        print(
            "Found these files/dirs in the model checkpoint directory: ",
            [x.name for x in params_folder.iterdir()],
        )

    # Just for demonstration: This is a user-implemented utility function.
    helper()

    # Iterate over subjects
    for subject in in_folder.iterdir():
        if subject.is_dir():
            print(f"Processing subject {subject.name}")
            pseudo_predict(subject, out_folder)

    print(
        "*** code execution finished:",
        application_name,
        "version:",
        application_version,
        "! ***",
    )

from argparse import ArgumentParser
from pathlib import Path
import warnings

import numpy as np
import SimpleITK as sitk


def dummy_predict(subject_dir: Path, output_dir: Path):
    # Please stick to this naming convention for your prediction!
    output_fname = output_dir / f"{subject_dir.name}_seg.nii.gz"

    # FeTS structure: one folder for each test case (subject), containing t1, t1ce, t2, flair
    image_files = list(subject_dir.glob('*.nii.gz'))
    if len(image_files) != 4:
        warnings.warn(f"Found {len(image_files)} files in subject directory, but expected four. Check data folder!!!")

    modalities = ["t1", "t1ce", "t2", "flair"]
    labels = [0, 1, 2, 4]
    seg_npy = None
    # In this dummy example, I "predict" class 0 from the t1-image, 1 from t1ce etc.,
    # using a simple thresholding operation
    for mod, lab in zip(modalities, labels):    
        img_path = next(subject_dir.glob(f'*_{mod}.nii.gz'))
        img_itk = sitk.ReadImage(str(img_path.absolute()))
        img_npy = sitk.GetArrayFromImage(img_itk)
        
        if seg_npy is None:
            seg_npy = np.zeros_like(img_npy)
        seg_npy[img_npy > np.percentile(img_npy, 95)] = lab
    
    seg_itk = sitk.GetImageFromArray(seg_npy)
    seg_itk.CopyInformation(img_itk)
    sitk.WriteImage(seg_itk, str(output_fname.absolute()))


if __name__ == "__main__":
    parser = ArgumentParser(description='Evaluate the model on the passed data folder')
    parser.add_argument('-i', '--in_folder', type=str,
                        help='Path to the data for which predictions are required.')
    parser.add_argument('-o', '--out_folder', type=str,
                        help='Path to the directory where segmentations should be saved.')
    args = parser.parse_args()

    in_folder = Path(args.in_folder)
    out_folder = Path(args.out_folder)
    params_folder = Path("/params")
    # no parameters are used in this example. Please copy model weights to the image when it is built.

    for subject in in_folder.iterdir():
        if subject.is_dir():
            print(f"Processing subject {subject.name}")
            dummy_predict(subject, out_folder)

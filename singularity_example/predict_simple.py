from argparse import ArgumentParser
from pathlib import Path
import warnings

import numpy as np
import SimpleITK as sitk


def predict_thresholding(subject_dir, output_dir, model="dummy"):
    output_fname = output_dir / f"{subject_dir.name}_{model}_seg.nii.gz"

    image_files = list(subject_dir.glob('*.nii.gz'))
    # subject dir is expected to have the four modalities
    if len(image_files) != 4:
        warnings.warn(f"Found {len(image_files)} files in subject directory, but expected four. Check data folder!!!")

    # use only t1 for "prediction"
    t1_img = next(subject_dir.glob('*_t1.nii.gz'))
    img_itk = sitk.ReadImage(str(t1_img.absolute()))
    img_npy = sitk.GetArrayFromImage(img_itk)
    seg_npy = np.zeros_like(img_npy)
    labels = [1, 2, 4]
    thresholds = np.percentile(img_npy[img_npy > 0.], [50, 70, 90])
    for thresh, lab in zip(thresholds, labels):
        seg_npy[img_npy > thresh] = lab
    
    seg_itk = sitk.GetImageFromArray(seg_npy)
    seg_itk.CopyInformation(img_itk)
    sitk.WriteImage(seg_itk, str(output_fname.absolute()))


if __name__ == "__main__":
    parser = ArgumentParser(description='Evaluate the model on the passed data folder')
    parser.add_argument('-i', '--in_folder', type=str,
                        help='Path to the data for which predictions are required.')
    parser.add_argument('-o', '--out_folder', type=str,
                        help='Path to the directory where segmentations should be saved.')
    parser.add_argument('-p', '--params_folder', type=str,
                        help='Path to saved model parameters.')

    args = parser.parse_args()

    in_folder = Path(args.in_folder)
    out_folder = Path(args.out_folder)
    params_folder = Path(args.params_folder)   # not used in this example

    # we need to figure out the case IDS in the folder
    for subject in in_folder.iterdir():
        if subject.is_dir():
            print(f"Processing subject {subject.name}")
            predict_thresholding(subject, out_folder)

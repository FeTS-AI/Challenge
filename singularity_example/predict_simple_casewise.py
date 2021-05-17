from argparse import ArgumentParser
from pathlib import Path
import warnings

import numpy as np
import SimpleITK as sitk

# TODO outdated; copy & adapt the predict_simple.py code once it is final.

def predict_thresholding_subject(subject_id, t1_path, t1c_path, t2_path, fl_path, output_dir, model="dummy"):
    output_fname = output_dir / f"{subject_id}_{model}_seg.nii.gz"

    # use only t1 for "prediction"
    img_itk = sitk.ReadImage(str(t1_path.absolute()))
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
    parser = ArgumentParser(
        description='Evaluate the model on the specified subject.')
    parser.add_argument('-s', '--subject_id', type=str,
                        help='Subjext ID.')
    parser.add_argument('-t1', '--t1_path', type=str,
                        help='absolute path to t1 image.')
    parser.add_argument('-t1c', '--t1c_path', type=str,
                        help='absolute path to t1 post contrast image.')
    parser.add_argument('-t2', '--t2_path', type=str,
                        help='absolute path to t2 image.')
    parser.add_argument('-fl', '--fl_path', type=str,
                        help='absolute path to flair image.')
    parser.add_argument('-o', '--out_folder', type=str,
                        help='absolute path to output directory where the container will write all results, ')
    parser.add_argument('-p', '--params_folder', type=str,
                        help='Path to saved model parameters.')

    args = parser.parse_args()

    subject_id = args.subject_id
    t1_path = Path(args.t1_path)
    t1c_path = Path(args.t1c_path)
    t2_path = Path(args.t2_path)
    fl_path = Path(args.fl_path)
    out_folder = Path(args.out_folder)
    params_folder = Path(args.params_folder)   # not used in this example

    # we need to figure out the case IDS in the folder
    print(f"Processing subject {subject_id}")
    predict_thresholding_subject(
        subject_id=subject_id,
        t1_path=t1_path,
        t1c_path=t1c_path,
        t2_path=t2_path,
        fl_path=fl_path,
        output_dir=out_folder
    )

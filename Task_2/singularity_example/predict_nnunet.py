from argparse import ArgumentParser
from pathlib import Path
import shutil

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.inference.ensemble_predictions import merge
from nnunet.inference.predict import predict_cases
import SimpleITK as sitk
import numpy as np


# part of custom segmentation post-processing1
def apply_brats_threshold(fname, output_fname, threshold, replace_with):
    img_itk = sitk.ReadImage(fname)
    img_npy = sitk.GetArrayFromImage(img_itk)
    num_enh = np.sum(img_npy == 3)
    if num_enh < threshold:
        print(fname, "had only %d enh voxels, those are now necrosis" % num_enh)
        img_npy[img_npy == 3] = replace_with
    img_itk_postprocessed = sitk.GetImageFromArray(img_npy)
    img_itk_postprocessed.CopyInformation(img_itk)
    sitk.WriteImage(img_itk_postprocessed, output_fname)


# nnUNet has a different label convention than BraTS; convert back here
def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg


def load_convert_save(filename):
    a = sitk.ReadImage(filename)
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, filename)


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Evaluate a nnunet model on the data in the input folder.')
    # Arguments which are part of the interface of your container
    parser.add_argument('-i', '--in_folder', type=str,
                        help='Path to the data for which predictions are required.')
    parser.add_argument('-o', '--out_folder', type=str,
                        help='Path to the directory where segmentations should be saved.')
    # Further arguments could be passed from your container runscript if necessary
    parser.add_argument('-p', '--params_folder', type=str,
                        help='Path to saved model parameters.')

    args = parser.parse_args()

    in_folder = args.in_folder
    out_folder = args.out_folder
    params_folder = args.params_folder

    threshold = 200
    # necrosis and non-enhancing tumor in nnUNet label convention
    # (postprocessing is applied before converting to brats labels!)
    replace_with = 2

    # this adds all available model paths (sub-dirs in params_folder) to a dict
    # each model has been trained on several folds with nnunet
    model_folds_dict = {}
    for model_path in Path(params_folder).iterdir():
        n_folds = len(list(model_path.glob("fold_*")))
        model_folds_dict[model_path] = tuple(range(n_folds))

    print("Found %d models in the parameter folder." % len(model_folds_dict))

    # figure out the case IDS in the folder
    case_identifiers = [p.name for p in Path(in_folder).iterdir() if p.is_dir()]
    print("Found %d case identifiers! Here is the list:\n%s" % (
        len(case_identifiers), '\n'.join(sorted(case_identifiers))))

    # Build list [[case1_t1, case1_t1ce, case1_t2, case1_flair],
    #             [case2_t1, case2_t1ce, case2_t2, case2_flair], ...] used by nnunet
    model_inputs_list = []
    for case in case_identifiers:
        t1_file = join(in_folder, case, case + "_brain_t1.nii.gz")
        t1c_file = join(in_folder, case, case + "_brain_t1ce.nii.gz")
        t2_file = join(in_folder, case, case + "_brain_t2.nii.gz")
        flair_file = join(in_folder, case, case + "_brain_flair.nii.gz")

        if not isfile(t1_file):
            print(f"file missing for case identifier {case}. Expected to find: {t1_file}")
        if not isfile(t1c_file):
            print(f"file missing for case identifier {case}. Expected to find: {t1c_file}")
        if not isfile(t2_file):
            print(f"file missing for case identifier {case}. Expected to find: {t2_file}")
        if not isfile(flair_file):
            print(f"file missing for case identifier {case}. Expected to find: {flair_file}")
        model_inputs_list.append([t1_file, t1c_file, t2_file, flair_file])

    # each model saves predictions in its own folder first; will be merged later
    tmp_prediction_dirs = []
    for model_path, folds in model_folds_dict.items():
        curr_out_folder = join(out_folder, model_path.name)
        tmp_prediction_dirs.append(curr_out_folder)
        maybe_mkdir_p(curr_out_folder)
        output_filenames = [
            # Please stick to this naming convention for your prediction!
            join(curr_out_folder, f"{case}_seg.nii.gz")
            for case in case_identifiers
        ]

        predict_cases(
            model=str(model_path),
            list_of_lists=model_inputs_list,
            output_filenames=output_filenames,
            folds=folds,
            save_npz=True,
            num_threads_preprocessing=6,
            num_threads_nifti_save=2,
            segs_from_prev_stage=None,
            do_tta=True,
            mixed_precision=True,
            overwrite_existing=True,
            all_in_gpu=False,
            step_size=0.5
        )

    merge(tmp_prediction_dirs, out_folder, 1, override=True,
          postprocessing_file=None, store_npz=False)

    for f in subfiles(out_folder):
        # custom post-processing of predicted segmentations
        apply_brats_threshold(f, f, threshold, replace_with)
        load_convert_save(f)

    # cleanup of temporary predictions
    _ = [shutil.rmtree(i) for i in tmp_prediction_dirs]

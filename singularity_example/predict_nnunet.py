# TODO:
# - refactor a bit
# - adapt to FeTS folder structure
# Feedback Fabian:
# wenn du das skript oeffentlich stellen willst macht es sinn
# gerade in predict_nnunet.py noch mal die variablennamen klarer zu machen
# und vielleicht mehr dokumentation zu schreiben

from argparse import ArgumentParser
from pathlib import Path
import shutil

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.inference.ensemble_predictions import merge
from nnunet.inference.predict import predict_cases
import SimpleITK as sitk
import numpy as np


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
    parser = ArgumentParser(description='Evaluate a nnunet model on the data in the input folder.')
    # Arguments which are part of the interface of your container
    parser.add_argument('-i', '--in_folder', type=str,
                        help='Path to the data for which predictions are required.')
    parser.add_argument('-o', '--out_folder', type=str,
                        help='Path to the directory where segmentations should be saved.')
    # Further arguments could be passed from your container runscript if necessary
    parser.add_argument('-p', '--params_folder', type=str,
                        help='Path to saved model parameters.')
    # TODO remove this argument in the final version
    parser.add_argument('--nnunet_naming', action='store_true',
                        help='Path to saved model parameters.')

    args = parser.parse_args()

    in_folder = args.in_folder
    out_folder = args.out_folder
    params_folder = args.params_folder
    
    algo_id = "nnunet"
    threshold = 200
    # necrosis and non-enhancing tumor in MY label convention (apply postprocessing before converting to brats labels!)
    replace_with = 2
    # TODO remove this and go with fets-naming; just for testing purposes
    nnunet_folder = args.nnunet_naming   # folder structure
    if nnunet_folder:
        t1_suffix = "_0001.nii.gz"
        t1ce_suffix = "_0002.nii.gz"
        t2_suffix = "_0003.nii.gz"
        flair_suffix = "_0000.nii.gz"
    else:
        t1_suffix = "_t1.nii.gz"
        t1ce_suffix = "_t1ce.nii.gz"
        t2_suffix = "_t2.nii.gz"
        flair_suffix = "_flair.nii.gz"

    # could include more models for ensembling
    model_list = [
        'nnUNetTrainerV2__nnUNetPlansv2.1',
    ]
    # model_list = ['nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5',
    #             'nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5',
    #             'nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5']
    folds_list = [tuple(np.arange(5)),]   # one for each model


    # we need to figure out the case IDS in the folder
    if nnunet_folder:
        case_identifiers = [filename[:-len(t1_suffix)]
                            for filename in subfiles(in_folder, suffix=t1_suffix, join=False)]
    else:
        case_identifiers = [p.name for p in Path(in_folder).iterdir() if p.is_dir()]
    print("Found %d case identifiers! Here is an example: %s" % (
        len(case_identifiers), np.random.choice(case_identifiers, replace=False)))

    # Build list [[case1_t1, case1_t1ce, case1_t2, case1_flair],
    #             [case2_t1, case2_t1ce, case2_t2, case2_flair], ...] used by nnunet
    model_inputs_list = []
    for case in case_identifiers:
        if nnunet_folder:
            t1_file = join(in_folder, case + t1_suffix)
            t1c_file = join(in_folder, case + t1ce_suffix)
            t2_file = join(in_folder, case + t2_suffix)
            flair_file = join(in_folder, case + flair_suffix)
        else:
            t1_file = join(in_folder, case, case + t1_suffix)
            t1c_file = join(in_folder, case, case + t1ce_suffix)
            t2_file = join(in_folder, case, case + t2_suffix)
            flair_file = join(in_folder, case, case + flair_suffix)

        if not isfile(t1_file):
            print("file missing for case identifier %s. Expected to find: %s" %
                (case, t1_file))
        if not isfile(t1c_file):
            print("file missing for case identifier %s. Expected to find: %s" %
                (case, t1c_file))
        if not isfile(t2_file):
            print("file missing for case identifier %s. Expected to find: %s" %
                (case, t2_file))
        if not isfile(flair_file):
            print("file missing for case identifier %s. Expected to find: %s" %
                (case, flair_file))
        model_inputs_list.append([t1_file, t1c_file, t2_file, flair_file])

    # each model saves predictions in its own folder first; will be merged later
    tmp_prediction_dirs = []
    for model_name, folds in zip(model_list, folds_list):
        curr_out_folder = join(out_folder, model_name)
        tmp_prediction_dirs.append(curr_out_folder)
        maybe_mkdir_p(curr_out_folder)
        curr_params_folder = join(params_folder, model_name)

        output_filenames = [join(curr_out_folder, f"{case}_{algo_id}_seg.nii.gz")
                            for case in case_identifiers]

        # TODO resolve this messy argument list
        predict_cases(curr_params_folder, model_inputs_list, output_filenames, folds, True, 6, 2, None, True, True,
                      True, False, 0.5)

    merge(tmp_prediction_dirs, out_folder, 1, True, None, False)

    for f in subfiles(out_folder):
        # custom post-processing of predicted segmentations
        apply_brats_threshold(f, f, threshold, replace_with)
        load_convert_save(f)

    # cleanup of temporary predictions
    _ = [shutil.rmtree(i) for i in tmp_prediction_dirs]

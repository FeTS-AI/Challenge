# TODO:
# - refactor a bit
# - adapt to FeTS folder structure

from argparse import ArgumentParser
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
    parser = ArgumentParser(description='Evaluate the model on the passed data folder')
    parser.add_argument('in_folder', type=str,
                        help='Path to the data for which predictions are required.')
    parser.add_argument('out_folder', type=str,
                        help='Path to the data for which predictions are required.')
    parser.add_argument('params_folder', type=str,
                        help='Path to saved model parameters.')
    parser.add_argument('--nnunet_naming', action='store_true',
                        help='Path to saved model parameters.')

    args = parser.parse_args()
    # params_folder = join(os.environ["RESULTS_FOLDER"], "nnUNet/2d/Task001_BrainTumour")
    in_folder = args.in_folder
    out_folder = args.out_folder
    params_folder = args.params_folder
    
    # TODO remove this and go with fets-naming; just for testing purposes
    if args.nnunet_naming:
        t1_suffix = "_0001.nii.gz"
        t1ce_suffix = "_0002.nii.gz"
        t2_suffix = "_0003.nii.gz"
        flair_suffix = "_0000.nii.gz"
    else:
        t1_suffix = "_t1.nii.gz"
        t1ce_suffix = "_t1ce.nii.gz"
        t2_suffix = "_t2.nii.gz"
        flair_suffix = "_flair.nii.gz"
    
    # necrosis and non-enhancing tumor in MY label convention (apply postprocessing before converting to brats labels!)
    replace_with = 2

    # could include more models for ensembling
    model_list = [
        'nnUNetTrainerV2__nnUNetPlansv2.1',
    ]
    # model_list = ['nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5',
    #             'nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5',
    #             'nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5']
    folds_list = [tuple(np.arange(5))]

    threshold = 200

    # we need to figure out the case IDS in the folder
    case_identifiers = [filename[:-len(t1_suffix)]
                        for filename in subfiles(in_folder, suffix=t1_suffix, join=False)]
    print("Found %d case identifiers! Here are some examples: %s" % (
        len(case_identifiers), np.random.choice(case_identifiers, replace=False)))

    list_of_lists = []
    for c in case_identifiers:
        t1_file = join(in_folder, c + t1_suffix)
        t1c_file = join(in_folder, c + t1ce_suffix)
        t2_file = join(in_folder, c + t2_suffix)
        flair_file = join(in_folder, c + flair_suffix)
        if not isfile(t1_file):
            print("file missing for case identifier %s. Expected to find: %s" %
                (c, t1_file))
        if not isfile(t1c_file):
            print("file missing for case identifier %s. Expected to find: %s" %
                (c, t1c_file))
        if not isfile(t2_file):
            print("file missing for case identifier %s. Expected to find: %s" %
                (c, t2_file))
        if not isfile(flair_file):
            print("file missing for case identifier %s. Expected to find: %s" %
                (c, flair_file))
        list_of_lists.append([t1_file, t1c_file, t2_file, flair_file])

    prediction_folders = []
    for model_name, folds in zip(model_list, folds_list):
        output_model = join(out_folder, model_name)
        prediction_folders.append(output_model)
        maybe_mkdir_p(output_model)
        params_folder_model = join(params_folder, model_name)

        output_filenames = [join(output_model, case + ".nii.gz")
                            for case in case_identifiers]

        predict_cases(params_folder_model, list_of_lists, output_filenames, folds, True, 6, 2, None, True, True,
                      True, False, 0.5)

    merge(prediction_folders, out_folder, 1, True, None, False)

    for f in subfiles(out_folder):
        apply_brats_threshold(f, f, threshold, replace_with)
        load_convert_save(f)

    _ = [shutil.rmtree(i) for i in prediction_folders]

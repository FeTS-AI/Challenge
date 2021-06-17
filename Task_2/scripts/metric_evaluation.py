# Adapted from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/evaluation/region_based_evaluation.py
import json
import logging
from multiprocessing.pool import Pool
from pathlib import Path

from medpy import metric
import SimpleITK as sitk
import numpy as np

MAX_HD95 = 200.


def create_region_from_mask(mask, join_labels: tuple):
    mask_new = np.zeros_like(mask, dtype=np.uint8)
    for l in join_labels:
        mask_new[mask == l] = 1
    return mask_new


def evaluate_case(file_pred: str, file_gt: str, regions):
    image_gt = sitk.GetArrayFromImage(sitk.ReadImage(file_gt))
    image_pred = sitk.GetArrayFromImage(sitk.ReadImage(file_pred))
    results = []
    for r in regions:
        mask_pred = create_region_from_mask(image_pred, r)
        mask_gt = create_region_from_mask(image_gt, r)
        dc = np.nan if np.sum(mask_gt) == 0 and np.sum(
            mask_pred) == 0 else metric.dc(mask_pred, mask_gt)
        if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0:
            hd95 = 0.
        elif np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0:
            # in the real evaluation this will be treated differently (last rank)
            hd95 = MAX_HD95
        else:
            hd95 = metric.hd95(mask_pred, mask_gt)
        results.append({'dice': dc, 'hausdorff': hd95})
    return results


def evaluate_regions(folder_predicted: str, folder_gt: str, processes=4, hausdorff=False, save_path=None):
    folder_predicted = Path(folder_predicted)
    folder_gt = Path(folder_gt)
    regions = {
        "whole_tumor": (1, 2, 4),
        "tumor_core": (1, 4),
        "enhancing_tumor": (4,)
    }
    region_names = list(regions)
    files_in_pred = [x.name for x in folder_predicted.iterdir() if x.name.endswith('.nii.gz')]
    files_in_gt = [x.name for x in folder_gt.iterdir() if x.name.endswith('.nii.gz')]
    # prediction and groundtruth should have the same filename (in different folders)
    have_no_gt = [i for i in files_in_pred if i not in files_in_gt]
    assert len(have_no_gt) == 0, "Some files in folder_predicted have no ground truth in folder_gt"
    have_no_pred = [i for i in files_in_gt if i not in files_in_pred]
    if len(have_no_pred) > 0:
        logging.warning(
            "Some files in folder_gt were not predicted (not present in folder_predicted)!")

    files_in_gt.sort()
    files_in_pred.sort()

    # run for all cases
    full_filenames_gt = [str(folder_gt / i) for i in files_in_pred]
    full_filenames_pred = [str(folder_predicted / i) for i in files_in_pred]

    p = Pool(processes)
    res = p.starmap(evaluate_case, zip(full_filenames_pred, full_filenames_gt, [
                    list(regions.values())] * len(files_in_pred)))
    p.close()
    p.join()

    # Rearrange results in json object
    json_obj = []
    for i in range(len(files_in_pred)):
        result_here = res[i]
        curr_dict = {'segmentation': files_in_pred[i]}
        for k, r in enumerate(region_names):
            curr_dict[f"{r}_dice"] = result_here[k]['dice']
            if hausdorff:
                curr_dict[f"{r}_hd95"] = result_here[k]['hausdorff']
        json_obj.append(curr_dict)
    # add empty fields for missing entries to json object
    for fname in have_no_pred:
        curr_dict = {'segmentation': fname}
        for k, r in enumerate(region_names):
            curr_dict[f"{r}_dice"] = 'n/a'
            if hausdorff:
                curr_dict[f"{r}_hd95"] = 'n/a'
        json_obj.append(curr_dict)
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(json_obj, f, indent=2)
    return json_obj

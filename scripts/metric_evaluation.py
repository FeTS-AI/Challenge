from copy import deepcopy
from multiprocessing.pool import Pool

from batchgenerators.utilities.file_and_folder_operations import *
from medpy import metric
import SimpleITK as sitk
import numpy as np
from nnunet.configuration import default_num_threads
from nnunet.postprocessing.consolidate_postprocessing import collect_cv_niftis

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
        dc = np.nan if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0 else metric.dc(mask_pred, mask_gt)
        if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0:
            hd95 = 0.
        elif np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0:
            hd95 = MAX_HD95   # in the real evaluation this will be treated differently (last rank)
        else:
            hd95 = metric.hd95(mask_pred, mask_gt)
        results.append({'dice': dc, 'hausdorff': hd95})
    return results


def evaluate_regions(folder_predicted: str, folder_gt: str, processes=default_num_threads):
    regions = {
        "whole tumor": (1, 2, 4),
        "tumor core": (1, 4),
        "enhancing tumor": (4,)
    }
    region_names = list(regions)
    files_in_pred = subfiles(folder_predicted, suffix='.nii.gz', join=False)
    files_in_gt = subfiles(folder_gt, suffix='.nii.gz', join=False)
    # prediction and groundtruth should have the same filename (in different folders)
    have_no_gt = [i for i in files_in_pred if i not in files_in_gt]
    assert len(have_no_gt) == 0, "Some files in folder_predicted have not ground truth in folder_gt"
    have_no_pred = [i for i in files_in_gt if i not in files_in_pred]
    if len(have_no_pred) > 0:
        print("WARNING! Some files in folder_gt were not predicted (not present in folder_predicted)!")

    files_in_gt.sort()
    files_in_pred.sort()

    # run for all cases
    full_filenames_gt = [join(folder_gt, i) for i in files_in_pred]
    full_filenames_pred = [join(folder_predicted, i) for i in files_in_pred]

    # res = []
    # for pred, gt in zip(full_filenames_pred, full_filenames_gt):
    #     res.append(evaluate_case(pred, gt, list(regions.values())))

    p = Pool(processes)
    res = p.starmap(evaluate_case, zip(full_filenames_pred, full_filenames_gt, [list(regions.values())] * len(files_in_pred)))
    p.close()
    p.join()

    # all_results = {r: [] for r in region_names}
    output_file = join(folder_predicted, 'summary.csv')
    print(output_file)
    with open(output_file, 'w') as f:
        # TODO maybe refactor
        f.write("casename")
        for r in region_names:
            for m in ['dc', 'hd95']:
                f.write(",%s" % (r + "_" + m))
        f.write("\n")
        for i in range(len(files_in_pred)):
            f.write(files_in_pred[i][:-7])   # .nii.gz
            result_here = res[i]
            for k, r in enumerate(region_names):
                dc = result_here[k]['dice']
                hd = result_here[k]['hausdorff']
                f.write(",%02.4f" % dc)
                f.write(",%02.4f" % hd)
                # all_results[r].append(dc)
            f.write("\n")

        # add empty fields for missing entries to csv
        for fname in have_no_pred:
            f.write(fname[:-7])   # .nii.gz
            for _ in region_names:
                f.write(",,")

        # f.write('mean')
        # for r in region_names:
        #     f.write(",%02.4f" % np.nanmean(all_results[r]))
        # f.write("\n")
        # f.write('median')
        # for r in region_names:
        #     f.write(",%02.4f" % np.nanmedian(all_results[r]))
        # f.write("\n")

        # f.write('mean (nan is 1)')
        # for r in region_names:
        #     tmp = np.array(all_results[r])
        #     tmp[np.isnan(tmp)] = 1
        #     f.write(",%02.4f" % np.mean(tmp))
        # f.write("\n")
        # f.write('median (nan is 1)')
        # for r in region_names:
        #     tmp = np.array(all_results[r])
        #     tmp[np.isnan(tmp)] = 1
        #     f.write(",%02.4f" % np.median(tmp))
        # f.write("\n")


if __name__ == '__main__':
    evaluate_regions('/home/m167k/Datasets/test_nnunet_brats/results/fets_test',
                     '/home/m167k/Datasets/test_nnunet_brats/labels_fets_official')
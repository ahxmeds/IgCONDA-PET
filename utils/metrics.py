import numpy as np 
import cc3d
from sklearn.metrics import precision_recall_curve, auc
from scipy.spatial.distance import cdist

#%%
def threshold_suvmax(ptarray):
    max_suv = np.max(ptarray)
    threshold_value = 0.41 * max_suv
    binary_mask = np.where(ptarray > threshold_value, 1, 0)
    return binary_mask

def convert_labels(mask):
    mask[mask == 2] = 1
    return mask 

def compute_slice_level_dice_score(truth, pred):
    truth_converted = convert_labels(truth)
    intersection = np.sum(truth_converted*pred)
    sum_ = np.sum(truth_converted) + np.sum(pred)
    dice_score = 2*intersection/sum_
    return dice_score

def compute_slice_level_fp_area(truth, pred):
    truth_converted = convert_labels(truth)
    intersection = np.sum(truth_converted*pred)    
    fp_area = np.sum(pred) - intersection
    return fp_area

def compute_slice_level_fn_area(truth, pred):
    truth_converted = convert_labels(truth)
    intersection = np.sum(truth_converted*pred)    
    fn_area = np.sum(truth) - intersection
    return fn_area

def compute_slice_level_iou(
    gtarray: np.ndarray,
    predarray: np.ndarray, 
) -> np.float64:
    gtarray = convert_labels(gtarray)
    intersection = np.sum(predarray[gtarray == 1])
    union = np.sum(gtarray) + np.sum(predarray) - intersection
    iou = intersection/union
    return iou

def compute_slice_level_sensitivity_criterion2(truth, pred):
    truth_converted = convert_labels(truth)
    gtarray_labeled_mask, num_lesions_gt = cc3d.connected_components(truth_converted, connectivity=4, return_N=True)
    predarray_labeled_mask, num_lesions_pred = cc3d.connected_components(pred, connectivity=4, return_N=True)
    gt_lesions_list = list(np.arange(1, num_lesions_gt+1))
    #initial values for TP, FP, FN
    TP = 0
    FP = 0 
    FN = num_lesions_gt 
    threshold = 0.5
    for i in range(1, num_lesions_pred+1):
        max_iou = 0
        match_gt_lesion = None 
        pred_lesion_mask = np.where(predarray_labeled_mask == i, 1, 0)
        for j in range(1, num_lesions_gt+1):
            gt_lesion_mask = np.where(gtarray_labeled_mask == j, 1, 0)
            iou = compute_slice_level_iou(gt_lesion_mask, pred_lesion_mask)
            if iou > max_iou:
                max_iou = iou
                match_gt_lesion = j
        if max_iou >= threshold:
            TP += 1
            gt_lesions_list.remove(match_gt_lesion)
        else:
            FP += 1
    FN = len(gt_lesions_list)
    
    sensitivity = TP/(TP + FN)
    return sensitivity

#%%
def is_suvmax_detected(
    gtarray: np.ndarray,
    predarray: np.ndarray,
    ptarray: np.ndarray,
) -> bool:
    gtarray = convert_labels(gtarray)
    prod = np.multiply(gtarray, ptarray)
    max_index = np.unravel_index(np.argmax(prod), prod.shape)
    if predarray[max_index] == 1:
        return True
    else:
        return False

#%%
def calculate_patient_level_tp_fp_fn(
    gtarray: np.ndarray,
    predarray: np.ndarray,
    criterion: str,
    threshold: np.float64 = None,
    ptarray: np.ndarray = None,
) -> (int, int, int):
    """Calculate patient-level TP, FP, and FN (for detection based metrics)
    via 3 criteria:

    criterion1: A predicted lesion is TP if any one of it's foreground voxels 
    overlaps with GT foreground. A predicted lesions that doesn't overlap with any 
    GT foreground is FP. As soon as a lesion is predicted as TP, it is removed
    from the set of GT lesions. The lesions that remain in the end in the GT lesions
    are FN. `criterion1` is the weakest detection criterion.

    criterion2: A predicted lesion is TP if more than `threshold`% of it's volume 
    overlaps with foreground GT. A predicted lesion is FP if it overlap fraction
    with foreground GT is between 0% and `threshold`%. As soon as a lesion is 
    predicted as TP, it is removed from the set of GT lesions. The lesions that 
    remain in the end in the GT lesions are FN. `criterion2` can be hard or weak 
    criterion based on the value of `threshold`.

    criterion3: A predicted lesion is TP if it overlaps with one the the GT lesion's 
    SUVmax voxel, hence this criterion requires the use of PET data (`ptarray`). A 
    predicted lesion that doesn't overlap with any GT lesion's SUVmax voxel is 
    considered FP. As soon as a lesion is predicted as TP, it is removed from the 
    set of GT lesions. The lesions that remain in the end in the GT lesions are FN. 
    `criterion3` is likely an easy criterion since a network is more likely to segment 
    high(er)-uptake regions`.

    Args:
        int (_type_): _description_
        int (_type_): _description_
        gtarray (_type_, optional): _description_. Defaults to None, ptarray: np.ndarray = None, )->(int.
    """
    gtarray = convert_labels(gtarray)
    gtarray_labeled_mask, num_lesions_gt = cc3d.connected_components(gtarray, connectivity=4, return_N=True)
    predarray_labeled_mask, num_lesions_pred = cc3d.connected_components(predarray, connectivity=4, return_N=True)
    gt_lesions_list = list(np.arange(1, num_lesions_gt+1))
    #initial values for TP, FP, FN
    TP = 0
    FP = 0 
    FN = num_lesions_gt 

    if criterion == 'criterion1':
        FN = 0 # for this criterion we are counting the number of FPs from 0 onwards, hence the reassignment
        for i in range(1, num_lesions_pred+1):
            pred_lesion_mask = np.where(predarray_labeled_mask == i, 1, 0)
            if np.any(pred_lesion_mask & (gtarray_labeled_mask > 0)):
                TP += 1
            else:
                FP += 1
        for j in range(1, num_lesions_gt+1):
            gt_lesion_mask = np.where(gtarray_labeled_mask == j, 1, 0)
            if not np.any(gt_lesion_mask & (predarray_labeled_mask > 0)):
                FN += 1

    elif criterion == 'criterion2':
        for i in range(1, num_lesions_pred+1):
            max_iou = 0
            match_gt_lesion = None 
            pred_lesion_mask = np.where(predarray_labeled_mask == i, 1, 0)
            for j in range(1, num_lesions_gt+1):
                gt_lesion_mask = np.where(gtarray_labeled_mask == j, 1, 0)
                iou = compute_slice_level_iou(gt_lesion_mask, pred_lesion_mask)
                if iou > max_iou:
                    max_iou = iou
                    match_gt_lesion = j
            if max_iou >= threshold:
                TP += 1
                gt_lesions_list.remove(match_gt_lesion)
            else:
                FP += 1
        FN = len(gt_lesions_list)

    elif criterion == 'criterion3':
        for i in range(1, num_lesions_pred+1):
            max_iou = 0
            match_gt_lesion = None
            pred_lesion_mask = np.where(predarray_labeled_mask == i, 1, 0)
            for j in range(1, num_lesions_gt+1):
                gt_lesion_mask = np.where(gtarray_labeled_mask == j, 1, 0)
                iou = compute_slice_level_iou(gt_lesion_mask, pred_lesion_mask)
                if iou > max_iou:
                    max_iou = iou 
                    match_gt_lesion = j
                    
            if max_iou == 0:
                FP += 1
            else:
                # match_gt_lesion has been defined with has the maximum iou with pred lesion i
                arr_gt_lesion = np.where(gtarray_labeled_mask == match_gt_lesion, 1, 0)
                if is_suvmax_detected(arr_gt_lesion, pred_lesion_mask, ptarray):
                    TP += 1
                    gt_lesions_list.remove(match_gt_lesion)
        FN = len(gt_lesions_list)

    else:
        print('Invalid criterion. Choose between criterion1, criterion2, or criterion3')
        return 
    
    return TP, FP, FN

#%%
def compute_auprc(
    gtarray, predarray_unbinarized
):
    gtarray_flat = gtarray.flatten()
    predarray_unbinarized_flat = predarray_unbinarized.flatten()
    precision, recall, _ = precision_recall_curve(gtarray_flat, predarray_unbinarized_flat)
    auprc = auc(recall, precision)
    return auprc


def compute_hd95(gtarray, predarray):
    """
    Compute the 95th percentile Hausdorff Distance (HD95) between two binary masks.

    Parameters:
        gt_mask (numpy.ndarray): Ground truth binary mask (2D array with 0s and 1s).
        pred_mask (numpy.ndarray): Predicted binary mask (2D array with 0s and 1s).

    Returns:
        float: The 95th percentile Hausdorff Distance.
    """
    # Get the coordinates of the 1s (foreground) in each mask
    gt_points = np.argwhere(gtarray == 1)
    pred_points = np.argwhere(predarray == 1)

    # Compute pairwise distances between ground truth and predicted points
    distances_gt_to_pred = cdist(gt_points, pred_points)
    distances_pred_to_gt = cdist(pred_points, gt_points)

    # Compute the closest distance for each point in both directions
    min_distances_gt_to_pred = np.min(distances_gt_to_pred, axis=1)
    min_distances_pred_to_gt = np.min(distances_pred_to_gt, axis=1)

    # Combine the distances
    all_distances = np.concatenate([min_distances_gt_to_pred, min_distances_pred_to_gt])

    # Compute the 95th percentile distance
    hd95 = np.percentile(all_distances, 95)

    return hd95


    
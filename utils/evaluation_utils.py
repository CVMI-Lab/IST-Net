# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Modified based on https://github.com/hughw19/NOCS_CVPR2019."""


import os
import sys
import numpy as np
import glob
import math
import _pickle as cPickle
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
from ctypes import *
import math

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """

    pre_shape = x.shape
    assert len(x.shape) == 2, x.shape
    new_x = x[~np.all(x == 0, axis=1)]
    post_shape = new_x.shape
    assert pre_shape[0] == post_shape[0]
    assert pre_shape[1] == post_shape[1]

    return new_x


def get_3d_bbox(scale, shift=0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                            [scale / 2, +scale / 2, -scale / 2],
                            [-scale / 2, +scale / 2, scale / 2],
                            [-scale / 2, +scale / 2, -scale / 2],
                            [+scale / 2, -scale / 2, scale / 2],
                            [+scale / 2, -scale / 2, -scale / 2],
                            [-scale / 2, -scale / 2, scale / 2],
                            [-scale / 2, -scale / 2, -scale / 2]]) + shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, RT):
    """
    Input:
        coordinates: [3, N]
        RT: [4, 4]
    Return
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones(
        (1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def compute_ap_from_matches_scores(pred_match, pred_scores, gt_match):
    # sort the scores from high to low
    # print(pred_match.shape, pred_scores.shape)
    assert pred_match.shape[0] == pred_scores.shape[0]

    score_indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[score_indices]
    pred_match = pred_match[score_indices]

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1])
                * precisions[indices])
    return ap


def compute_3d_iou_new(RT_1, RT_2, scales_1, scales_2, handle_visibility, class_name_1, class_name_2):
    '''Computes IoU overlaps between two 3d bboxes.
       bbox_3d_1, bbox_3d_1: [3, 8]
    '''
    # flatten masks
    def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
        noc_cube_1 = get_3d_bbox(scales_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + \
            np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    if RT_1 is None or RT_2 is None:
        return -1

    symmetry_flag = False
    if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or (class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility == 0):
        # print('*'*10)

        noc_cube_1 = get_3d_bbox(scales_1, 0)
        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        def y_rotation_matrix(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                             [0, 1, 0, 0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [0, 0, 0, 1]])

        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = RT_1@y_rotation_matrix(2*math.pi*i/float(n))
            max_iou = max(max_iou,
                          asymmetric_3d_iou(rotated_RT_1, RT_2, scales_1, scales_2))
    else:
        max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)

    return max_iou


def compute_combination_RT_degree_cm_symmetry(RT_1, RT_2, scale, class_id, handle_visibility, synset_names):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter


    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]

    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    '''

    # make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    # symmetric when rotating around y-axis
    if synset_names[class_id] in ['bottle', 'can', 'bowl']:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    # symmetric when rotating around y-axis
    elif synset_names[class_id] == 'mug' and handle_visibility == 0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) / scale
    result = np.array([theta, shift])

    return result


def compute_combination_3d_matches(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                                   pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales,
                                   iou_3d_thresholds, degree_thesholds, shift_thesholds, score_threshold=0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]

        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    RT_overlaps = np.zeros((num_pred, num_gt, 2), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            # overlaps[i, j] = compute_3d_iou(pred_3d_bboxs[i], gt_3d_bboxs[j], gt_handle_visibility[j],
            #    synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])
            overlaps[i, j] = compute_3d_iou_new(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j],
                                                gt_handle_visibility[j], synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])

            RT_overlaps[i, j, :] = compute_combination_RT_degree_cm_symmetry(pred_RTs[i], gt_RTs[j], np.cbrt(
                np.linalg.det(gt_RTs[j, :3, :3])), gt_class_ids[j], gt_handle_visibility[j], synset_names)

    # Loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    num_degree_thes = len(degree_thesholds)
    num_shift_thes = len(shift_thesholds)
    pred_matches = -1 * \
        np.ones([num_degree_thes, num_shift_thes, num_iou_3d_thres, num_pred])
    gt_matches = -1 * \
        np.ones([num_degree_thes, num_shift_thes, num_iou_3d_thres, num_gt])

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for d, degree_thres in enumerate(degree_thesholds):
            for t, shift_thres in enumerate(shift_thesholds):
                for i in range(len(pred_boxes)):
                    # Find best matching ground truth box
                    # 1. Sort matches by score
                    sorted_ixs_by_iou = np.argsort(overlaps[i])[::-1]
                    # 2. Remove low scores
                    low_score_idx = np.where(
                        overlaps[i, sorted_ixs_by_iou] < score_threshold)[0]
                    if low_score_idx.size > 0:
                        sorted_ixs_by_iou = sorted_ixs_by_iou[:low_score_idx[0]]
                    # 3. Find the match
                    for j in sorted_ixs_by_iou:
                        if gt_matches[d, t, s, j] > -1:
                            continue
                        # If we reach IoU smaller than the threshold, end the loop
                        iou = overlaps[i, j]
                        r_error = RT_overlaps[i, j, 0]
                        t_error = RT_overlaps[i, j, 1]

                        if iou < iou_thres or r_error > degree_thres or t_error > shift_thres:
                            break

                        if not pred_class_ids[i] == gt_class_ids[j]:
                            continue

                        if iou >= iou_thres or r_error <= degree_thres or t_error <= shift_thres:
                            gt_matches[d, t, s, j] = i
                            pred_matches[d, t, s, i] = j
                            break

    return gt_matches, pred_matches, indices


def compute_combination_mAP(final_results, synset_names, degree_thresholds=[5, 10, 15], shift_thresholds=[0.1, 0.2], iou_3d_thresholds=[0.1]):
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)

    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)

    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    aps = np.zeros((num_classes + 1, num_degree_thres,
                    num_shift_thres, num_iou_thres))
    pred_matches_all = [np.zeros(
        (num_degree_thres, num_shift_thres, num_iou_thres, 0)) for _ in range(num_classes)]
    gt_matches_all = [np.zeros(
        (num_degree_thres, num_shift_thres, num_iou_thres, 0)) for _ in range(num_classes)]
    pred_scores_all = [np.zeros(
        (num_degree_thres, num_shift_thres, num_iou_thres, 0)) for _ in range(num_classes)]

    for progress, result in tqdm(enumerate(final_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_RTs = np.array(result['gt_RTs'])
        gt_scales = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']

        pred_bboxes = np.array(result['pred_bboxes'])
        pred_class_ids = result['pred_class_ids']
        pred_scales = result['pred_scales']
        pred_scores = result['pred_scores']
        pred_RTs = np.array(result['pred_RTs'])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros(0)
            cls_gt_scales = gt_scales[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 3))
            cls_gt_RTs = gt_RTs[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 4, 4))

            cls_pred_class_ids = pred_class_ids[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_bboxes = pred_bboxes[pred_class_ids == cls_id, :] if len(
                pred_class_ids) else np.zeros((0, 4))
            cls_pred_scores = pred_scores[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_scales = pred_scales[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 3))

            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids == cls_id] if len(
                    gt_class_ids) else np.ones(0)

            gt_match, pred_match, pred_indiced = compute_combination_3d_matches(cls_gt_class_ids, cls_gt_RTs, cls_gt_scales, cls_gt_handle_visibility, synset_names,
                                                                                cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
                                                                                iou_thres_list, degree_thres_list, shift_thres_list)
            if len(pred_indiced):
                cls_pred_class_ids = cls_pred_class_ids[pred_indiced]
                cls_pred_RTs = cls_pred_RTs[pred_indiced]
                cls_pred_scores = cls_pred_scores[pred_indiced]
                cls_pred_bboxes = cls_pred_bboxes[pred_indiced]

            pred_matches_all[cls_id] = np.concatenate(
                (pred_matches_all[cls_id], pred_match), axis=-1)
            cls_pred_scores_tile = np.tile(
                cls_pred_scores, (num_degree_thres, num_shift_thres, num_iou_thres, 1))
            pred_scores_all[cls_id] = np.concatenate(
                (pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert pred_matches_all[cls_id].shape[-1] == pred_scores_all[cls_id].shape[-1]
            gt_matches_all[cls_id] = np.concatenate(
                (gt_matches_all[cls_id], gt_match), axis=-1)

    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        for s, iou_thres in enumerate(iou_thres_list):
            for d, degree_thres in enumerate(degree_thres_list):
                for t, shift_thres in enumerate(shift_thres_list):
                    aps[cls_id, d, t, s] = compute_ap_from_matches_scores(pred_matches_all[cls_id][d, t, s, :],
                                                                          pred_scores_all[cls_id][d,
                                                                                                  t, s, :],
                                                                          gt_matches_all[cls_id][d, t, s, :])
    # for i in range(6):
    #     print(i+1)
    #     print('IoU75, 5  degree,  5% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(5), shift_thres_list.index(0.05), iou_thres_list.index(0.75)]*100))
    #     print('IoU75, 10 degree,  5% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(10), shift_thres_list.index(0.05), iou_thres_list.index(0.75)]*100))
    #     print('IoU75, 5  degree, 10% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(5), shift_thres_list.index(0.10), iou_thres_list.index(0.75)]*100))
    #     print('IoU50, 5  degree, 20% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(5), shift_thres_list.index(0.20), iou_thres_list.index(0.50)]*100))
    #     print('IoU50, 10 degree, 10% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(10), shift_thres_list.index(0.10), iou_thres_list.index(0.50)]*100))
    #     print('IoU50, 10 degree, 20% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(10), shift_thres_list.index(0.20), iou_thres_list.index(0.50)]*100))

    # print('ALL:')
    aps[-1, :, :, :] = np.mean(aps[1:-1, :, :, :], axis=0)

    print('IoU75, 5  degree,  5% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(5), shift_thres_list.index(0.05), iou_thres_list.index(0.75)]*100))
    print('IoU75, 10 degree,  5% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(10), shift_thres_list.index(0.05), iou_thres_list.index(0.75)]*100))
    print('IoU75, 5  degree, 10% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(5), shift_thres_list.index(0.10), iou_thres_list.index(0.75)]*100))
    print('IoU50, 5  degree, 20% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(5), shift_thres_list.index(0.20), iou_thres_list.index(0.50)]*100))
    print('IoU50, 10 degree, 10% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(10), shift_thres_list.index(0.10), iou_thres_list.index(0.50)]*100))
    print('IoU50, 10 degree, 20% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(10), shift_thres_list.index(0.20), iou_thres_list.index(0.50)]*100))

    return aps


def compute_3d_matches_for_each_gt(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                       pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales):
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]

        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    overlaps = np.zeros((num_gt, num_pred), dtype=np.float32)
    # print("num_gt:", num_gt, "num_pred:", num_pred)
    for j in range(num_gt):
        for i in range(num_pred):
            # overlaps[i, j] = compute_3d_iou(pred_3d_bboxs[i], gt_3d_bboxs[j], gt_handle_visibility[j],
            #    synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])
            overlaps[j, i] = compute_3d_iou_new(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j],
                                                gt_handle_visibility[j], synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])


    # num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_pred, ])
    gt_matches = -1 * np.ones([num_gt,], dtype=np.int32)

    for i in range(num_gt):
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        
        for j in sorted_ixs:
            if pred_matches[j] > -1:
                continue

            if not pred_class_ids[j] == gt_class_ids[i]:
                continue

            gt_matches[i] = j
            pred_matches[j] = i
            break

    return gt_matches, indices




def compute_3d_matches(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                       pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales,
                       iou_3d_thresholds, score_threshold=0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]

        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            # overlaps[i, j] = compute_3d_iou(pred_3d_bboxs[i], gt_3d_bboxs[j], gt_handle_visibility[j],
            #    synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])
            overlaps[i, j] = compute_3d_iou_new(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j],
                                                gt_handle_visibility[j], synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])

    # Loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(len(pred_boxes)):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(
                overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                #print('gt_match: ', gt_match[j])
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                #print('iou: ', iou)
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue

                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break

    return gt_matches, pred_matches, overlaps, indices


def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id, handle_visibility, synset_names):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter


    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]

    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    '''

    # make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    # symmetric when rotating around y-axis
    if synset_names[class_id] in ['bottle', 'can', 'bowl']:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    # symmetric when rotating around y-axis
    elif synset_names[class_id] == 'mug' and handle_visibility == 0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result


def compute_RT_overlaps(gt_class_ids, gt_RTs, gt_handle_visibility,
                        pred_class_ids, pred_RTs,
                        synset_names):
    """Finds overlaps between prediction and ground truth instances.
    Returns:
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # print('num of gt instances: {}, num of pred instances: {}'.format(len(gt_class_ids), len(gt_class_ids)))
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_degree_cm_symmetry(pred_RTs[i],
                                                              gt_RTs[j],
                                                              gt_class_ids[j],
                                                              gt_handle_visibility[j],
                                                              synset_names)

    return overlaps


def compute_match_from_degree_cm(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)

    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2

    for d, degree_thres in enumerate(degree_thres_list):
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Remove low scores
                # low_score_idx = np.where(sum_degree_shift >= 100)[0]
                # if low_score_idx.size > 0:
                #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
                # 3. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    #print(j, len(gt_match), len(pred_class_ids), len(gt_class_ids))
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue

                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches


def compute_independent_mAP(final_results, synset_names, degree_thresholds=[360], shift_thresholds=[100], iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=True, logger=None, plot_figure=True, log_dir=None):

    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)

    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)

    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 0))
                            for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 0))
                           for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 0))
                          for _ in range(num_classes)]

    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)]
    pose_gt_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)]
    pose_pred_scores_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)]

    # loop over results to gather pred matches and gt matches for iou and pose metrics
    progress = 0
    for progress, result in tqdm(enumerate(final_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_RTs = np.array(result['gt_RTs'])
        gt_scales = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']

        pred_bboxes = np.array(result['pred_bboxes'])
        pred_class_ids = result['pred_class_ids']
        pred_scales = result['pred_scales']
        pred_scores = result['pred_scores']
        pred_RTs = np.array(result['pred_RTs'])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros(0)
            cls_gt_scales = gt_scales[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 3))
            cls_gt_RTs = gt_RTs[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 4, 4))

            # ipdb.set_trace()
            cls_pred_class_ids = pred_class_ids[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_bboxes = pred_bboxes[pred_class_ids == cls_id, :] if len(
                pred_class_ids) else np.zeros((0, 4))
            cls_pred_scores = pred_scores[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_scales = pred_scales[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 3))

            # calculate the overlap between each gt instance and pred instance
            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids == cls_id] if len(
                    gt_class_ids) else np.ones(0)

            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = compute_3d_matches(cls_gt_class_ids, cls_gt_RTs, cls_gt_scales, cls_gt_handle_visibility, synset_names,
                                                                                           cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
                                                                                           iou_thres_list)

            
            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
                cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]

            # print("iou_pred_matches_all shape:", iou_pred_matches_all[cls_id].shape, "iou_cls_pred_match shape:", iou_cls_pred_match.shape)
            # assert False
            iou_pred_matches_all[cls_id] = np.concatenate(
                (iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1)
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            iou_pred_scores_all[cls_id] = np.concatenate(
                (iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]
            iou_gt_matches_all[cls_id] = np.concatenate(
                (iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1)

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)

                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]

                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_bboxes = cls_pred_bboxes[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros((0, 4))

                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(
                    iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1] if len(
                    iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(
                    iou_thres_gt_match) > 0 else np.zeros(0)

            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_RTs, cls_gt_handle_visibility,
                                              cls_pred_class_ids, cls_pred_RTs,
                                              synset_names)

            pose_cls_gt_match, pose_cls_pred_match = compute_match_from_degree_cm(RT_overlaps,
                                                                                  cls_pred_class_ids,
                                                                                  cls_gt_class_ids,
                                                                                  degree_thres_list,
                                                                                  shift_thres_list)

            pose_pred_matches_all[cls_id] = np.concatenate(
                (pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1)

            cls_pred_scores_tile = np.tile(
                cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            pose_pred_scores_all[cls_id] = np.concatenate(
                (pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert pose_pred_scores_all[cls_id].shape[2] == pose_pred_matches_all[cls_id].shape[2], '{} vs. {}'.format(
                pose_pred_scores_all[cls_id].shape, pose_pred_matches_all[cls_id].shape)
            pose_gt_matches_all[cls_id] = np.concatenate(
                (pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1)


    # draw iou 3d AP vs. iou thresholds
    fig_iou = plt.figure(figsize=(30,10))
    # ax_iou = plt.subplot(111)
    ax_iou = plt.subplot(131)
    plt.ylabel('AP')
    plt.ylim((0, 1))
    plt.tick_params(labelsize=20)
    plt.xlabel('3D IoU thresholds', fontsize=24)

    iou_dict = {}
    iou_dict['thres_list'] = iou_thres_list
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        for s, iou_thres in enumerate(iou_thres_list):
            iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
                                                                   iou_pred_scores_all[cls_id][s, :],
                                                                   iou_gt_matches_all[cls_id][s, :])
        ax_iou.plot(iou_thres_list, iou_3d_aps[cls_id, :], label=class_name)

    iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-1, :], axis=0)
    ax_iou.plot(iou_thres_list, iou_3d_aps[-1, :], label='mean')
    iou_dict['aps'] = iou_3d_aps

    for i, degree_thres in enumerate(degree_thres_list):
        for j, shift_thres in enumerate(shift_thres_list):
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

                pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(cls_pose_pred_matches_all,
                                                                        cls_pose_pred_scores_all,
                                                                        cls_pose_gt_matches_all)

            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])

    ax_trans = plt.subplot(132)
    plt.ylim((0, 1))
    plt.tick_params(labelsize=20)
    plt.xlabel('Rotation/degree', fontsize=24)
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        # print(class_name)
        ax_trans.plot(
            degree_thres_list[:-1], pose_aps[cls_id, :-1, -1], label=class_name)
    
    ax_trans.plot(degree_thres_list[:-1], pose_aps[-1, :-1, -1], label='mean')
    # pose_dict['aps'] = pose_aps

    ax_rot = plt.subplot(133)
    plt.ylim((0, 1))
    plt.tick_params(labelsize=20)
    plt.xlabel('translation/cm', fontsize=24)
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        # print(class_name)
        ax_rot.plot(shift_thres_list[:-1],
                    pose_aps[cls_id, -1, :-1], label=class_name)

    ax_rot.plot(shift_thres_list[:-1], pose_aps[-1, -1, :-1], label='mean')

    plt.legend(loc='lower right')

    plot_save_path = os.path.join(log_dir, 'visual')
    if not os.path.isdir(plot_save_path):
        os.mkdir(plot_save_path)

    output_path = os.path.join(
        plot_save_path, 'mAP_{}-{}cm.png'.format(shift_thres_list[0], shift_thres_list[-2]))
    ax_rot.legend()

    if plot_figure:
        fig_iou.savefig(output_path)
    plt.close(fig_iou)

    if logger is not None:
        logger.warning('3D IoU at 25: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.25)] * 100))
        logger.warning('3D IoU at 50: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.5)] * 100))
        logger.warning('3D IoU at 75: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.75)] * 100))

        logger.warning('5 degree, 2cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(5), shift_thres_list.index(2)] * 100))
        logger.warning('5 degree, 5cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(5), shift_thres_list.index(5)] * 100))

        logger.warning('10 degree, 2cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(2)] * 100))
        logger.warning('10 degree, 5cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(5)] * 100))
        logger.warning('10 degree, 10cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(10)] * 100))
        
        # print per class iou
        logger.warning('####### Per Class result ###################')
        for idx in range(1, len(synset_names)):
            logger.warning('category {}'.format(synset_names[idx]))
            logger.warning('mAP:')
            logger.warning('3D IoU at 25: {:.1f}'.format(iou_3d_aps[idx, iou_thres_list.index(0.25)] * 100))
            logger.warning('3D IoU at 50: {:.1f}'.format(iou_3d_aps[idx, iou_thres_list.index(0.5)] * 100))
            logger.warning('3D IoU at 75: {:.1f}'.format(iou_3d_aps[idx, iou_thres_list.index(0.75)] * 100))
            logger.warning('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(5), shift_thres_list.index(2)] * 100))
            logger.warning('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(5), shift_thres_list.index(5)] * 100))
            logger.warning('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(10), shift_thres_list.index(2)] * 100))
            logger.warning('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(10), shift_thres_list.index(5)] * 100))
            logger.warning('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(10), shift_thres_list.index(10)] * 100))
    else:
        print('3D IoU at 25: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.25)] * 100))
        print('3D IoU at 50: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.5)] * 100))
        print('3D IoU at 75: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.75)] * 100))

        print('5 degree, 2cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(5), shift_thres_list.index(2)] * 100))
        print('5 degree, 5cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(5), shift_thres_list.index(5)] * 100))

        print('10 degree, 2cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(2)] * 100))
        print('10 degree, 5cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(5)] * 100))
        print('10 degree, 10cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(10)] * 100))

        # print per class iou
        print('####### Per Class result ###################')
        for idx in range(1, len(synset_names)):
            print('category {}'.format(synset_names[idx]))
            print('mAP:')
            print('3D IoU at 25: {:.1f}'.format(iou_3d_aps[idx, iou_thres_list.index(0.25)] * 100))
            print('3D IoU at 50: {:.1f}'.format(iou_3d_aps[idx, iou_thres_list.index(0.5)] * 100))
            print('3D IoU at 75: {:.1f}'.format(iou_3d_aps[idx, iou_thres_list.index(0.75)] * 100))
            print('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(5), shift_thres_list.index(2)] * 100))
            print('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(5), shift_thres_list.index(5)] * 100))
            print('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(10), shift_thres_list.index(2)] * 100))
            print('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(10), shift_thres_list.index(5)] * 100))
            print('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_thres_list.index(10), shift_thres_list.index(10)] * 100))

    return iou_3d_aps, pose_aps


def evaluate(path, logger=None):
    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug']  # 6

    result_pkl_list = glob.glob(os.path.join(path, 'results*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    print('image num: {}'.format(len(result_pkl_list)))

    final_results = []
    count = 0
    for pkl_path in result_pkl_list:
        # print(pkl_path)
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if not 'gt_handle_visibility' in result:
                result['gt_handle_visibility'] = np.ones_like(
                    result['gt_class_ids'])
                print('can\'t find gt_handle_visibility in the pkl.')
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])

        if type(result) is list:
            final_results += result
        elif type(result) is dict:
            final_results.append(result)
        else:
            assert False
        count+=1
        # if count > 500:
        #     break
    # print("Compute combined mAP: ")
    # compute_combination_mAP(final_results, synset_names,
    #                         degree_thresholds=[5, 10, 20],
    #                         shift_thresholds=[0.05, 0.1, 0.2],
    #                         iou_3d_thresholds=[0.25, 0.50, 0.75])

    print("Compute independent mAP: ")
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    compute_independent_mAP(final_results, synset_names,
                            degree_thresholds=degree_thres_list,
                            shift_thresholds=shift_thres_list,
                            iou_3d_thresholds=iou_thres_list, logger=logger, log_dir=path)

    # compute_independent_mAP(final_results, synset_names,
    #                         degree_thresholds=[5, 10],
    #                         shift_thresholds=[2, 5, 10],
    #                         iou_3d_thresholds=[0.10, 0.25, 0.50, 0.75], logger=logger, log_dir=path)

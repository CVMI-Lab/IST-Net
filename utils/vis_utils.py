import logging
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
from tqdm import tqdm


def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def get_3d_bbox(scale, shift = 0):
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
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def draw(img, imgpts, axes, color):
    imgpts = np.int32(imgpts).reshape(-1, 2)


    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 3)


    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 3)

    
    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)


    # draw axes
    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3) ## y last


    return 


def draw_text(draw_image, bbox, text, draw_box=False):
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    thickness = 1
    

    retval, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    
    bbox_margin = 10
    text_margin = 10
    
    text_box_pos_tl = (min(bbox[1] + bbox_margin, 635 - retval[0] - 2* text_margin) , min(bbox[2] + bbox_margin, 475 - retval[1] - 2* text_margin)) 
    text_box_pos_br = (text_box_pos_tl[0] + retval[0] + 2* text_margin,  text_box_pos_tl[1] + retval[1] + 2* text_margin)

    # text_pose is the bottom-left corner of the text
    text_pos = (text_box_pos_tl[0] + text_margin, text_box_pos_br[1] - text_margin - 3)
    
    if draw_box:
        cv2.rectangle(draw_image, 
                      (bbox[1], bbox[0]),
                      (bbox[3], bbox[2]),
                      (255, 0, 0), 2)

    cv2.rectangle(draw_image, 
                  text_box_pos_tl,
                  text_box_pos_br,
                  (255,0,0), -1)
    
    cv2.rectangle(draw_image, 
                  text_box_pos_tl,
                  text_box_pos_br,
                  (0,0,0), 1)

    cv2.putText(draw_image, text, text_pos,
                fontFace, fontScale, (255,255,255), thickness)

    return draw_image


def align_rotation(sRT):
    """ Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """
    s = np.cbrt(np.linalg.det(sRT[:3, :3]))
    R = sRT[:3, :3] / s
    T = sRT[:3, 3]

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                      [0.0,            1.0,  0.0           ],
                      [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    rotation = R @ s_map
    aligned_sRT = np.identity(4, dtype=np.float32)
    aligned_sRT[:3, :3] = s * rotation
    aligned_sRT[:3, 3] = T
    return aligned_sRT


def draw_bboxes(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
    # draw pillars in minor darker color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)

    return img


def draw_detections(img, out_dir, data_name, img_id, intrinsics, pred_sRT, pred_size, pred_class_ids,
                    gt_sRT, gt_size, gt_class_ids, nocs_sRT, nocs_size, nocs_class_ids, draw_gt=True, draw_nocs=True):
    """ Visualize pose predictions.
    """
    out_path = os.path.join(out_dir, '{}_{}_pred.png'.format(data_name, img_id))

    # draw nocs results - BLUE color
    if draw_nocs:
        for i in range(nocs_sRT.shape[0]):
            if nocs_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(nocs_sRT[i, :, :])
            else:
                sRT = nocs_sRT[i, :, :]
            bbox_3d = get_3d_bbox(nocs_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (255, 0, 0))
    # darw ground truth - GREEN color
    if draw_gt:
        for i in range(gt_sRT.shape[0]):
            if gt_class_ids[i] in [1, 2, 4]:
                sRT = align_rotation(gt_sRT[i, :, :])
            else:
                sRT = gt_sRT[i, :, :]
            bbox_3d = get_3d_bbox(gt_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (0, 255, 0))
            # assert False
    # darw prediction - RED color
    for i in range(pred_sRT.shape[0]):
        if pred_class_ids[i] in [1, 2, 4]:
            sRT = align_rotation(pred_sRT[i, :, :])
        else:
            sRT = pred_sRT[i, :, :]
        bbox_3d = get_3d_bbox(pred_size[i, :], 0)
        transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        img = draw_bboxes(img, projected_bbox, (0, 0, 255))

    cv2.imwrite(out_path, img)
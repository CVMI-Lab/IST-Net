import os
import cv2
import numpy as np
import collections

def load_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

def load_composed_depth(img_path):
    """ Load depth image from img_path. """
    img_path_ = img_path.replace('/data/CAMERA/', '/data/camera_full_depths/')
    depth_path = img_path_ + '_composed.png'
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, -1)
        if len(depth.shape) == 3:
            # This is encoded depth image, let's convert
            # NOTE: RGB is actually BGR in opencv
            depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
            depth16 = np.where(depth16==32001, 0, depth16)
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'
        return depth16
    else:
        return None


def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def fill_in_multiscale_old(depth_map, max_depth=3.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 1.5 < depths < 3.0 m
        dilation_kernel_med: dilation kernel to use for 0.75 < depths < 1.5 m
        dilation_kernel_near: dilation kernel to use for 0.01 < depths < 0.75 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.01) & (depths_in <= 0.75)
    valid_pixels_med = (depths_in > 0.75) & (depths_in <= 1.5)
    valid_pixels_far = (depths_in > 1.5)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.01)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.01)
    valid_pixels_med = (dilated_med > 0.01)
    valid_pixels_far = (dilated_far > 0.01)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.01)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.01)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.01)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.01, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.01) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.01) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.01) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.01)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict


def fill_in_multiscale(depth_map, max_depth=8.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.01) & (depths_in <= 1.0)
    valid_pixels_med = (depths_in > 1.0) & (depths_in <= 2.0)
    valid_pixels_far = (depths_in > 2.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.01)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.01)
    valid_pixels_med = (dilated_med > 0.01)
    valid_pixels_far = (dilated_far > 0.01)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.01)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.01)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.01)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.01, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.01) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.01) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.01) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.01)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict

# vim: ts=4 sw=4 sts=4 expandtab

def fill_missing(
        dpt, cam_scale, scale_2_80m, fill_type='multiscale',
        extrapolate=False, show_process=False, blur_type='bilateral'
):
    dpt = dpt / cam_scale * scale_2_80m
    projected_depth = dpt.copy()
    if fill_type == 'fast':
        final_dpt = fill_in_fast(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            # max_depth=2.0
        )
    elif fill_type == 'multiscale':
        final_dpt, process_dict = fill_in_multiscale(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            show_process=show_process,
            max_depth=3.0
        )
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    dpt = final_dpt / scale_2_80m * cam_scale
    return dpt
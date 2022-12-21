import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import PurePath
from tqdm import tqdm
import random


def pixel_transform(pixel_a, parameters_a, parameters_b):
    """
        Converts pixel coordinates to equivalent ones for a camera with different parameters.
    """

    row_quotient = (pixel_a[0] - parameters_a['cx']) / parameters_a['fx']
    column_quotient = (pixel_a[1] - parameters_a['cy']) / parameters_a['fy']

    b_row = row_quotient * parameters_b['fx'] + parameters_b['cx']
    b_column = column_quotient * parameters_b['fy'] + parameters_b['cy']

    return int(b_row), int(b_column)


def align_depth(depth, camera_params):
    """
        Aligns a depth map to the camera described by camera_params.
    """

    depth[depth == 0] = np.nan

    depth_parameters = camera_params['depth']
    image_parameters = camera_params['color']

    top_left = pixel_transform((0, 0), image_parameters, depth_parameters)

    if min(top_left) < 0:
        padding = int(2*(np.ceil(-min(top_left))))
        depth_parameters['cx'] += padding
        depth_parameters['cy'] += padding
        padded_depth = np.pad(depth, padding, 'constant', constant_values=[np.nan, np.nan])
        top_left = pixel_transform((0, 0), image_parameters, depth_parameters)
        assert min(top_left) >= 0

    else:
        padded_depth = depth

    bottom_right = (image_parameters['height'] - 1, image_parameters['width'] - 1)
    bottom_right = pixel_transform(bottom_right, image_parameters, depth_parameters)
    aligned_depth = padded_depth[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    aligned_depth = cv2.resize(aligned_depth, dsize=(image_parameters['width'], image_parameters['height']))

    return aligned_depth


def camera_parameters(datapath):
    """
        Reads camera parameters from 12Scenes scene directory.
    """

    with open(os.path.join(datapath, 'info.txt'), 'r') as f:
        lines = f.readlines()
    color_width = int([line for line in lines if 'm_colorWidth' in line][0].split(' = ')[1])
    color_height = int([line for line in lines if 'm_colorHeight' in line][0].split(' = ')[1])
    color_intrinsics = [line for line in lines if 'm_calibrationColorIntrinsic' in line][0]
    color_intrinsics = color_intrinsics.split(' = ')[1]
    color_intrinsics = color_intrinsics.split(' ')
    color_fx = float(color_intrinsics[0])
    color_cx = float(color_intrinsics[2])
    color_fy = float(color_intrinsics[5])
    color_cy = float(color_intrinsics[6])

    depth_width = int([line for line in lines if 'm_depthWidth' in line][0].split(' = ')[1])
    depth_height = int([line for line in lines if 'm_depthHeight' in line][0].split(' = ')[1])
    depth_intrinsics = [line for line in lines if 'm_calibrationDepthIntrinsic' in line][0]
    depth_intrinsics = depth_intrinsics.split(' = ')[1]
    depth_intrinsics = depth_intrinsics.split(' ')
    depth_fx = float(depth_intrinsics[0])
    depth_cx = float(depth_intrinsics[2])
    depth_fy = float(depth_intrinsics[5])
    depth_cy = float(depth_intrinsics[6])

    parameter_dict = {'color': {'cx': color_cx, 'cy': color_cy, 'fx': color_fx, 'fy': color_fy, 'height': color_height,
                                'width': color_width},
                      'depth': {'cx': depth_cx, 'cy': depth_cy, 'fx': depth_fx, 'fy': depth_fy, 'height': depth_height,
                                'width': depth_width}}

    return parameter_dict


def downsample_image_dir(in_dir, factor, out_dir, filter=lambda x: True):
    """
        Downsamples all depth maps in in_dir by factor and saves them in out_dir.
    """
    img_fnames = [fname for fname in os.listdir(in_dir) if filter(fname)]
    for img_fname in tqdm(img_fnames, desc=f'Downsampling images with factor {factor}.'):
        if not filter(img_fname):
            continue
        img = cv2.imread(os.path.join(in_dir, img_fname))
        new_dims = [int(dim / factor) for dim in img.shape[:2]]
        new_dims.reverse()
        downsampled_img = cv2.resize(img, new_dims)
        save_path = os.path.join(out_dir, img_fname)
        cv2.imwrite(save_path, downsampled_img)
        pass

    return


def downsample_depth_dir(in_dir, factor, out_dir, filter=lambda x: 'depth.npy' in x):
    """
        Downsamples all depth maps in in_dir by factor and saves them in out_dir.
    """
    fnames = [fname for fname in os.listdir(in_dir) if filter(fname)]
    for fname in tqdm(fnames, desc=f'Downsampling depth with factor {factor}.'):

        with open(os.path.join(in_dir, fname), 'rb') as f:
            depth = np.load(f, allow_pickle=True)

        new_dims = [int(dim / factor) for dim in depth.shape]
        new_dims.reverse()
        downsampled_depth = cv2.resize(depth, new_dims)
        save_path = os.path.join(out_dir, fname)
        with open(save_path, 'wb') as f:
            np.save(f, downsampled_depth, allow_pickle=True)

    return


def get_radial_depth_multiplier(camera_params):
    """
        Point-wise multiplication matrix that converts the distance from (xy-) plane to distance from (focal) point.
    """

    fx = camera_params['fx']
    fy = camera_params['fy']
    assert max(fx / fy, fy / fx) < 1.01
    f = fx
    distance_from_center = np.array([[[column - camera_params['cx'], row - camera_params['cy'], f]
                                      for column in range(camera_params['width'])] for row in
                                     range(camera_params['height'])])
    radius = np.linalg.norm(distance_from_center, axis=2)
    radial_depth_multiplier = radius / f
    return radial_depth_multiplier


def align_depth_dir(in_dir,
                    out_dir,
                    camera_params,
                    selected_frames,
                    to_radial_depth):
    """
        Aligns all depth maps in in_dir to the camera described by camera_params and saves them in out_dir.
    """

    frames = selected_frames

    if to_radial_depth:
        radial_depth_multiplier = get_radial_depth_multiplier(camera_params['depth'])

    for frame in frames:
        path = os.path.join(in_dir, f'frame-{frame}.depth.png')
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = np.array(depth, dtype="float")
        if to_radial_depth:
            depth = np.multiply(depth, radial_depth_multiplier)

        aligned_depth = align_depth(depth, camera_params)
        save_name = os.path.join(out_dir, f'frame-{frame}.depth.npy')
        with open(save_name, 'wb') as f:
            np.save(f, aligned_depth, allow_pickle=True)

    return


def prepare_dataset(datapath, downsample_factors=[2, 4, 8, 16], n_frames=10, seed=0, to_radial_depth=True):
    """
        Puts a 12Scenes scene directory into the format needed by our Nerfstudio dataparsers.
    """

    # Sample subset of frames that will be included in the training / evaluation process (others will be ignored)
    frames = [fname[6:12] for fname in os.listdir(os.path.join(datapath, 'data')) if 'color.jpg' in fname]
    frames = list(np.unique(frames))
    if n_frames == -1:
        selected_frames = frames
    else:
        random.seed(seed)
        selected_frames = random.sample(frames, n_frames)

    with open(os.path.join(datapath, 'selected_frames.txt'), 'w') as f:
        f.write('\n'.join(selected_frames))

    # Read camera parameters
    camera_params = camera_parameters(datapath)

    # Create dirs
    for factor in downsample_factors:
        os.makedirs(os.path.join(datapath, f'images_{factor}'), exist_ok=True)
        os.makedirs(os.path.join(datapath, f'depth_{factor}'), exist_ok=True)
    os.makedirs(os.path.join(datapath, f'depth_1'), exist_ok=True)

    # Downsample images
    for factor in downsample_factors:
        downsample_image_dir(in_dir=os.path.join(datapath, 'data'),
                             factor=factor,
                             out_dir=os.path.join(datapath, f'images_{factor}'),
                             filter=(lambda fname: 'color.jpg' in fname and fname[6:12] in selected_frames))

    # Depth alignment scale 1
    align_depth_dir(in_dir=os.path.join(datapath, 'data'),
                    out_dir=os.path.join(datapath, 'depth_1'),
                    camera_params=camera_params,
                    selected_frames=selected_frames,
                    to_radial_depth=to_radial_depth)

    # Down sample depth images
    for factor in downsample_factors:
        downsample_depth_dir(in_dir=os.path.join(datapath, 'depth_1'),
                            factor=factor,
                            out_dir=os.path.join(datapath, f'depth_{factor}'),
                            filter=(lambda fname: 'depth.npy' in fname and fname[6:12] in selected_frames))

    return

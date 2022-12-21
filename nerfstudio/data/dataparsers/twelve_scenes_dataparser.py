
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
# from nerfstudio.utils.io import load_from_json

import os

# import random

# random.seed(0)

# N_FRAMES = 100

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class TwelveScenesDataParserConfig(DataParserConfig):
    """TwelveScenes dataset config"""

    _target: Type = field(default_factory=lambda: TwelveScenes)
    """target class to instantiate"""
    data: Path = Path("data/apt1/kitchen")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "up"
    """The method to use for orientation."""
    center_poses: bool = True
    """Whether to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""


@dataclass
class TwelveScenes(DataParser):
    """TwelveScene DatasetParser"""

    config: TwelveScenesDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        print('Dataparsing start.')

        image_filenames = []
        mask_filenames = []
        poses = []
        num_skipped_image_filenames = 0

        with open(os.path.join(self.config.data, 'selected_frames.txt'), 'r') as f:
            frames = f.read().splitlines()

        for frame in frames:
            filepath = PurePath(os.path.join('data', f'frame-{frame}.color.jpg'))
            fname = self._get_fname(filepath)

            image_filenames.append(fname)

            with open(os.path.join(self.config.data, 'data', f'frame-{frame}.pose.txt'), 'r') as f:
                lines = f.read().splitlines()
            lines = [[float(value) for value in line.split(' ')] for line in lines]
            pose = np.array(lines)
            pose[:3,:3] = np.matmul(pose[:3,:3], np.diag([1, -1, -1]))

            poses.append(pose)


            if "mask_path" in frame:
                mask_filepath = PurePath(frame["mask_path"])
                mask_fname = self._get_fname(mask_filepath, downsample_folder_prefix="masks_")
                mask_filenames.append(mask_fname)
        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_percentage)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_poses=self.config.center_poses,
        )[0]

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(poses[:, :3, 3]))

        poses[:, :3, 3] *= scale_factor * self.config.scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        camera_type = CameraType.PERSPECTIVE

        # Read camera parameters from 12Scenes scene dict
        with open(os.path.join(self.config.data, 'info.txt'), 'r') as f:
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

        distortion_params = camera_utils.get_distortion_params(
            k1=0.0,
            k2=0.0,
            k3=0.0,
            k4=0.0,
            p1=0.0,
            p2=0.0,
        )

        # Set cameras
        cameras = Cameras(
            fx=color_fx,
            fy=color_fy,
            cx=color_cx,
            cy=color_cy,
            distortion_params=distortion_params,
            height=color_height,
            width=color_width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        # Adjust camera parameters to downscale factor
        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
        )

        print('Dataparsing end.')

        return dataparser_outputs

    def _get_fname(self, filepath: PurePath, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxillary image data, e.g. masks
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(self.config.data / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (self.config.data / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return self.config.data / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return self.config.data / filepath

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class CUSTOMDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(CUSTOMDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class CUSTOMRAWDataset(CUSTOMDataset):
    def __init__(self, *args, **kwargs):
        super(CUSTOMRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "FrameBuffer_{:04d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "data", f_str)

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "Depth_{:04d}.png".format(frame_index)

        depth_path = os.path.join(
            self.data_path,
            folder,
            "data/groundtruth",
            f_str)

        depth_gt = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt



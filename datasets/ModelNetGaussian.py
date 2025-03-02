"""
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
"""

import os
import torch
import torch.utils.data as data
import numpy as np

# --- For normal execution, import the following modules
from .io import IO
from .build import DATASETS
from utils.logger import *
import math

# --- For debugging, import the following modules
# import sys
# import pdb
# sys.path.append('/home/bin_ren/projects/gaussian/gaussian_mae/base_mae')
# from datasets.io import IO
# from datasets.build import DATASETS
# from utils.logger import *


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def read_gaussian_attribute(vertex, attribute):
    assert "xyz" in attribute, "At least need xyz attribute"
    if "xyz" in attribute:
        x = vertex["x"].astype(np.float32)
        y = vertex["y"].astype(np.float32)
        z = vertex["z"].astype(np.float32)
        data = np.stack((x, y, z), axis=-1)  # [n, 3]

    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
        opacity = np_sigmoid(opacity)
        # opacity range from 0 to 1
        data = np.concatenate((data, opacity), axis=-1)
        # print("data", data.shape)

    if "scale" in attribute and "rotation" in attribute:
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)

        scales = np.exp(scales)  # scale normalization

        # print("scales", scales.min(), scales.max())

        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        # always set the first to be positive
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]

        data = np.concatenate((data, scales, rots), axis=-1)

    if "sh" in attribute:
        # sphere homrincals to rgb
        features_dc = np.zeros((data.shape[0], 3, 1))
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)

        feature_pc = features_dc.reshape(-1, 3)

        data = np.concatenate((data, feature_pc), axis=1)

    return data


@DATASETS.register_module()
class ModelNetGaussian(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.gs_path = config.GS_PATH
        self.num_category = config.NUM_CATEGORY
        self.attribute = config.ATTRIBUTE
        self.subset = config.subset
        self.norm_attribute = config.norm_attribute
        print("======================")
        print("config", config)

        if self.num_category == 10:
            self.catfile = os.path.join(self.data_root, "modelnet10_shape_names.txt")
            if self.subset == "train":
                self.data_list_file = os.path.join(
                    self.data_root, "modelnet10_train.txt"
                )
            else:
                self.data_list_file = os.path.join(
                    self.data_root, "modelnet10_test.txt"
                )

        else:
            self.catfile = os.path.join(self.data_root, "modelnet40_shape_names.txt")
            if self.subset == "train":
                self.data_list_file = os.path.join(
                    self.data_root, "modelnet40_train.txt"
                )
            else:
                self.data_list_file = os.path.join(
                    self.data_root, "modelnet40_test.txt"
                )

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.sample_points_num = config.N_POINTS

        print_log(
            f"[DATASET] Using Guassian Attribute {self.attribute}",
            logger="ModelNet Gaussian",
        )
        print_log(
            f"[DATASET] sample out {self.sample_points_num} points",
            logger="ModelNet Gaussian",
        )
        print_log(
            f"[DATASET] Open file {self.data_list_file}", logger="ModelNet Gaussian"
        )
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            # print("line", line)
            taxonomy_id = line.split("_")[:-1]
            taxonomy_id = "_".join(taxonomy_id)
            model_id = line
            file_path = os.path.join(
                self.gs_path, taxonomy_id, self.subset, line, "point_cloud.ply"
            )
            cls = self.classes[taxonomy_id]
            self.file_list.append(
                {
                    "taxonomy_id": taxonomy_id,
                    "model_id": model_id,
                    "file_path": file_path,
                    "cls": cls,
                }
            )
        print_log(
            f"[DATASET] {len(self.file_list)} instances were loaded",
            logger="ModelNet Gaussian",
        )

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def pc_norm_gs(self, pc, attribute=["xyz"]):
        """pc: NxC, return NxC"""
        pc_xyz = pc[..., :3]
        centroid = np.mean(pc_xyz, axis=0)
        pc_xyz = pc_xyz - centroid
        m = np.max(np.sqrt(np.sum(pc_xyz**2, axis=1)))
        pc_xyz = pc_xyz / m
        # inside a sphere
        pc[..., :3] = pc_xyz
        pc[..., 4:7] = pc[..., 4:7] / m

        if "opacity" in attribute:
            # normalize to a -1 to 1 range
            min_opacity = 0
            max_opacity = 1
            pc[..., 3] = (pc[..., 3] - min_opacity) / (
                max_opacity - min_opacity
            ) * 2 - 1

            # print("opc", pc[..., 3].min(), pc[..., 3].max())

        if "scale" in attribute:
            # normalize to a -1 to 1 range
            s_center = np.mean(pc[..., 4:7], axis=0)
            pc[..., 4:7] = pc[..., 4:7] - s_center
            s_m = np.max(np.sqrt(np.sum(pc[..., 4:7] ** 2, axis=1)))
            pc[..., 4:7] = pc[..., 4:7] / s_m
            # print("scale", pc[..., 4:7].min(), pc[..., 4:7].max())
        else:
            s_center = np.zeros(3)
            s_m = 1

        if "sh" in attribute:
            sh = pc[..., 11:14]
            sh = sh * 0.28209479177387814
            sh = np.clip(sh, -0.5, 0.5)
            sh = 2 * sh / math.sqrt(3)
            pc[..., 11:14] = sh

        return pc, s_center, s_m

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        label = sample["cls"]
        try:
            gs = IO.get(os.path.join(self.gs_path, sample["file_path"]))
        except Exception:
            print("Error in loading", os.path.join(self.gs_path, sample["file_path"]))

        vertex = gs["vertex"]

        data = read_gaussian_attribute(vertex, self.attribute)
        data, s_center, s_m = self.pc_norm_gs(data, self.norm_attribute)

        choice_gs = np.random.choice(len(data), self.sample_points_num, replace=True)
        data = data[choice_gs, :]

        data = torch.from_numpy(data).float()

        # print("dataloader", data.shape)
        return "ModelNetGS", sample["file_path"], (data, label)

    def __len__(self):
        return len(self.file_list)

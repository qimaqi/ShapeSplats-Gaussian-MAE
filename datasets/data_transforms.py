import numpy as np
import torch
import random


class PointcloudRotate(object):
    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array(
                [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
            )
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
            pc[i, :, :] = torch.matmul(pc[i], R)
        return pc


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2.0 / 3.0, scale_high=3.0 / 2.0, translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def augument(self, pc, attribute=["xyz"]):
        bsize = pc.size()[0]
        feature_dim = pc.size()[-1]
        if ["xyz"] not in attribute:
            return pc

        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(
                low=-self.translate_range, high=self.translate_range, size=[3]
            )
            pc[i, :, 0:3] = (
                torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
                + torch.from_numpy(xyz2).float().cuda()
            )
        if "scale" in attribute:
            pc[..., 4:7] = torch.mul(
                pc[..., 4:7], torch.from_numpy(xyz1).float().cuda()
            )

        return pc

    # def __call__(self, pc, scale_c=None, scale_m=None, attribute=['xyz']):
    #     bsize = pc.size()[0]
    #     feature_dim = pc.size()[-1]
    #     if ['xyz'] not in attribute:
    #         return pc, scale_c, scale_m
    #     else:
    #         new_centers = torch.zeros(bsize, 3)
    #         new_scales = torch.zeros(bsize)

    #         for i in range(bsize):
    #             xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
    #             xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
    #             pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
    #         if 'scale' in attribute:
    #             if scale_c is not None and scale_m is not None:
    #                 pc[...,4:7] = pc[...,4:7] * scale_m.unsqueeze(-1).unsqueeze(-1).to(pc.device)  + scale_c.unsqueeze(1).repeat(1,pc.shape[1],1).to(pc.device)
    #             pc[...,4:7] = torch.mul(pc[...,4:7], torch.from_numpy(xyz1).float().cuda())

    #             if scale_c is not None and scale_m is not None:
    #                 new_center = torch.mean(pc[i, :, 4:7], dim=0)
    #                 pc[i, :, 4:7] = pc[i, :, 4:7] - new_center
    #                 new_scale = torch.max(torch.sqrt(torch.sum(pc[i, :, 4:7]**2, dim=1)))
    #                 pc[i, :, 4:7] = pc[i, :, 4:7] / new_scale

    #                 new_centers[i] = new_center
    #                 new_scales[i] = new_scale

    #     return pc, new_centers, new_scales


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = (
                pc.new(pc.size(1), 3)
                .normal_(mean=0.0, std=self.std)
                .clamp_(-self.clip, self.clip)
            )
            pc[i, :, 0:3] += jittered_data

        return pc


class PointcloudScale(object):
    def __init__(self, scale_low=2.0 / 3.0, scale_high=3.0 / 2.0):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])

            pc[i, :, 0:3] = torch.mul(
                pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()
            )

        return pc


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(
                low=-self.translate_range, high=self.translate_range, size=[3]
            )

            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()

        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(
                    len(drop_idx), 1
                )  # set to the first point
                pc[i, :, :] = cur_pc

        return pc


class RandomHorizontalFlip(object):

    def __init__(self, upright_axis="z", is_temporal=False):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {"x": 0, "y": 1, "z": 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords):
        bsize = coords.size()[0]
        for i in range(bsize):
            if random.random() < 0.95:
                for curr_ax in self.horz_axes:
                    if random.random() < 0.5:
                        coord_max = torch.max(coords[i, :, curr_ax])
                        coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
        return coords

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from utils import misc
from utils.logger import *
from knn_cuda import KNN

from models.neural_nn import N3AggregationBase


class Encoder(nn.Module):  # Embedding module
    def __init__(self, encoder_channel, attribute=["xyz"]):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.attribute = attribute
        input_dim = 3

        if "opacity" in attribute:
            input_dim += 1
        if "sh" in attribute:
            input_dim += 3
        if "scale" in attribute:
            input_dim += 3
        if "rotation" in attribute:
            input_dim += 4

        # print("input_dim", input_dim)
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G M 3, gs B G M K
        B: Batch size
        G: Number of groups
        O: Number of potential neighbors per point
        M: Number of points per group
        K: Number of dimensions per point (depends on attributes)
        -----------------
        feature_global : B G C
        """
        bs, g, n, _ = point_groups.shape
        # choose the attribute we want
        # print("point_groups", point_groups.shape)
        attribute_index = [0, 1, 2]
        if "opacity" in self.attribute:
            opacity_index = [3]
            attribute_index.extend(opacity_index)
        if "scale" in self.attribute:
            scale_index = [4, 5, 6]
            attribute_index.extend(scale_index)
        if "rotation" in self.attribute:
            rotation_index = [7, 8, 9, 10]
            attribute_index.extend(rotation_index)
        if "sh" in self.attribute:
            sh_index = [11, 12, 13]
            attribute_index.extend(sh_index)

        # choose the attribute we want
        point_groups = point_groups[..., attribute_index]

        point_groups = point_groups.reshape(bs * g, n, -1).contiguous()  # (BG, K, M)
        # idx_tensor = torch.tensor(attribute_index, device=point_groups.device).unsqueeze(0).int().repeat(bs*g, 1)
        # point_groups = pointnet2_utils.gather_operation(point_groups, idx_tensor)

        # encoder
        # print("point_groups", point_groups.shape)
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 m
        # print("feature",  feature.shape)  ([8192, 256, 32])
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        # print("feature_global",  feature_global.shape) ([8192, 256, 1])
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # BG 512 m
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class SoftEncoder(nn.Module):
    def __init__(self, encoder_channel, k=32, attribute=["xyz"], temp_opt={}):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.attribute = attribute
        self.k = k
        # self.group_size = group_size
        # self.num_group = num_group

        input_dim = 3
        if "opacity" in attribute:
            input_dim += 1
        if "sh" in attribute:
            input_dim += 3
        if "scale" in attribute:
            input_dim += 3
        if "rotation" in attribute:
            input_dim += 4

        # Initial convolution layers for embedding
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.encoder_channel, 1),
        )

        # NeuralNearestNeighbors and aggregation
        self.n3_agg = N3AggregationBase(k, temp_opt=temp_opt)

    def forward(self, point_groups):
        """
        Args:
            point_groups: Tensor of shape (B, G, O, K)
        Returns:
            z: Tensor of shape (B, G, C), where C is the dimension of the encoded token feature
        """
        bs, g, o, k = point_groups.shape
        point_groups = point_groups.reshape(bs * g, o, k).contiguous()

        attribute_index = [0, 1, 2]
        if "opacity" in self.attribute:
            opacity_index = [3]
            attribute_index.extend(opacity_index)
        if "scale" in self.attribute:
            scale_index = [4, 5, 6]
            attribute_index.extend(scale_index)
        if "rotation" in self.attribute:
            rotation_index = [7, 8, 9, 10]
            attribute_index.extend(rotation_index)
        if "sh" in self.attribute:
            sh_index = [11, 12, 13]
            attribute_index.extend(sh_index)

        point_groups = point_groups[..., attribute_index]
        point_groups = point_groups.contiguous().reshape(
            bs * g, o, len(attribute_index)
        )

        # Tokenization via first convolution layer
        feature = self.first_conv(point_groups.transpose(2, 1))  # (BG, E, O)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (BG, E, 1)

        # Preparing for N3 aggregation
        xe = feature  # (BG, E, O), use as database items
        ye = feature_global  # (BG, E, 1), use as query items

        # Perform N3 aggregation
        # (BG, E, k), k is the neighbor num in the hard knn setting
        z = self.n3_agg(xe, ye)
        # TODO, could try then use the same soft knn again to find 1 neighbor out of k, essentially aggerate to group feature

        z = torch.cat(
            [torch.max(z, dim=2, keepdim=True)[0].repeat(1, 1, self.k), z], dim=1
        )  # append the max pooling to local feature
        z = self.second_conv(z)  # (BG, C, k)
        z_global = torch.max(z, dim=2, keepdim=False)[0]  # (BG, C)

        # output: (B, G, C)
        return z_global.reshape(bs, g, -1)

    def compute_indices(self, features):
        """
        Compute indices for potential neighbors
        Args:
            features: Tensor of shape (B, G, 512)
        Returns:
            I: Index tensor of shape (B, G, O)
        """
        B, G, _ = features.shape
        _, indices = torch.topk(features, self.group_size, dim=2)
        indices = indices.unsqueeze(2).expand(B, G, self.group_size, -1)
        return indices


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size, attribute=["xyz"], soft_knn=False):
        super().__init__()
        self.num_group = num_group
        self.group_size = (
            group_size if not soft_knn else int(1.25 * group_size)
        )  # expand potential neighbor size if soft_knn
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.attribute = attribute
        attribute_index = []
        if "xyz" in self.attribute:
            xyz_index = [0, 1, 2]
            attribute_index.extend(xyz_index)
        if "opacity" in self.attribute:
            opacity_index = [3]
            attribute_index.extend(opacity_index)
        if "scale" in self.attribute:
            scale_index = [4, 5, 6]
            attribute_index.extend(scale_index)
        if "rotation" in self.attribute:
            rotation_index = [7, 8, 9, 10]
            attribute_index.extend(rotation_index)
        if "sh" in self.attribute:
            sh_index = [11, 12, 13]
            attribute_index.extend(sh_index)

        self.attribute_index = attribute_index

    def forward(self, xyz):
        """
        input: B N 3 or B N K
        ---------------------------
        output: B G M 3 or (B G O 3) if select potential neighbors
        center : B G 3
        """
        batch_size, num_points, feature_dim = xyz.shape
        if feature_dim == 3:  # only xyz
            # this is for pointcloud implementation, ignore
            # fps the centers out
            center = misc.fps(xyz, self.num_group)  # B G 3
            # knn to get the neighborhood
            _, idx = self.knn(xyz, center)  # B G M
            assert idx.size(1) == self.num_group
            assert idx.size(2) == self.group_size
            idx_base = (
                torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1)
                * num_points
            )
            idx = idx + idx_base
            idx = idx.view(-1)
            neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
            neighborhood = neighborhood.view(
                batch_size, self.num_group, self.group_size, 3
            ).contiguous()
            # normalize
            neighborhood = neighborhood - center.unsqueeze(2)

        else:  # gaussian attribute
            center = misc.fps_gs(xyz, self.num_group, self.attribute)  # B G K
            center_group = center[..., self.attribute_index]
            xyz_group = xyz[..., self.attribute_index]
            _, idx = self.knn(xyz_group, center_group)  # B G M

            assert idx.size(1) == self.num_group
            assert idx.size(2) == self.group_size
            idx_base = (
                torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1)
                * num_points
            )
            idx = idx + idx_base
            idx = idx.view(-1)
            neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
            neighborhood = neighborhood.view(
                batch_size, self.num_group, self.group_size, -1
            ).contiguous()
            neighborhood[..., :3] = neighborhood[..., :3] - center.unsqueeze(2)[..., :3]

        return neighborhood, center


# Transformers
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, list)
                        else drop_path_rate
                    ),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, list)
                        else drop_path_rate
                    ),
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        # only return the mask tokens predict pixel
        x = self.head(self.norm(x[:, -return_token_num:]))
        return x

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .build import MODELS
from utils.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)
from utils.logger import print_log
import random
from knn_cuda import KNN
from models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    Encoder,
    Group,
    SoftEncoder,
)
from pytorch3d.loss import chamfer_distance


# pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f"[args] {config.transformer_config}", logger="Transformer")
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = (
            Encoder(encoder_channel=self.encoder_dims, attribute=config.attribute)
            if not kwargs.get("soft_knn", False)
            else SoftEncoder(
                encoder_channel=self.encoder_dims, attribute=config.attribute
            )
        )

        self.mask_type = config.transformer_config.mask_type
        self.group_attribite = config.group_attribute
        self.norm_attribute = config.norm_attribute

        self.pos_feature_dim = []
        if "xyz" in config.group_attribute:
            self.pos_feature_dim.extend([0, 1, 2])

        if "opacity" in config.group_attribute:
            self.pos_feature_dim.append(3)

        if "scale" in config.group_attribute:
            self.pos_feature_dim.extend([4, 5, 6])

        if "rotation" in config.group_attribute:
            self.pos_feature_dim.extend([7, 8, 9, 10])

        if "sh" in config.group_attribute:
            # here we use only first 3 sh, since most dataset color is simple
            self.pos_feature_dim.extend([11, 12, 13])

        self.pos_embed = nn.Sequential(
            nn.Linear(len(self.pos_feature_dim), 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        """
        center : B G 3
        --------------
        mask : B G (bool)
        """
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(
                points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1
            )  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(B, G, dtype=torch.bool, device=center.device)

        self.num_mask = int(self.mask_ratio * G)
        rand_indices = torch.rand(B, G, device=center.device)
        _, selected_indices = torch.topk(
            rand_indices, self.num_mask, dim=1, largest=False
        )
        overall_mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
        overall_mask.scatter_(1, selected_indices, True)

        return overall_mask

    def forward(self, neighborhood, center, noaug=False):
        """

        Args:
            neighborhood (_type_): B G O K, O: number of potential neighbors, e.g. 256, K: dimension of gaussian feature
            center (_type_): _description_
            noaug (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # generate mask
        if self.mask_type == "rand":
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        # B G C, C: dimension of token feature
        group_input_tokens = self.encoder(neighborhood)
        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(
            batch_size, -1, C
        )  # B, 0.4G, C

        masked_center = center[~bool_masked_pos].reshape(
            batch_size, -1, len(self.pos_feature_dim)
        )
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


# pretrain model
@MODELS.register_module()
class Gaussian_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log("[Gaussian_MAE] ", logger="Gaussian_MAE")
        self.config = config
        self.soft_knn = getattr(config, "soft_knn", False)
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config, soft_knn=self.soft_knn)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.knn = KNN(k=config.group_size, transpose_mode=True)
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.attribute = config.attribute
        self.group_attribute = config.group_attribute
        self.norm_attribute = config.norm_attribute

        self.pos_feature_dim = []
        if "xyz" in config.group_attribute:
            self.pos_feature_dim.extend([0, 1, 2])

        if "opacity" in config.group_attribute:
            self.pos_feature_dim.append(3)

        if "scale" in config.group_attribute:
            self.pos_feature_dim.extend([4, 5, 6])

        if "rotation" in config.group_attribute:
            self.pos_feature_dim.extend([7, 8, 9, 10])

        if "sh" in config.group_attribute:
            self.pos_feature_dim.extend([11, 12, 13])

        print("pos embedding size", self.pos_feature_dim)
        print("group_attribute", self.group_attribute)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(len(self.pos_feature_dim), 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)
        ]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(
            f"[Gaussian_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
            logger="Gaussian_MAE",
        )
        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
            attribute=config.group_attribute,
            soft_knn=self.soft_knn,
        )

        # prediction head for xyz
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        # predication head for density
        if "opacity" in self.attribute:
            self.opacity_head = nn.Sequential(
                nn.Conv1d(self.trans_dim, 1 * self.group_size, 1),
                (
                    nn.Sigmoid() if "opacity" not in self.norm_attribute else nn.Tanh()
                ),  # otherwise the opacity range is [-1, 1]
            )

        if "scale" in self.attribute and "rotation" in self.attribute:
            self.scale_head = nn.Sequential(
                nn.Conv1d(self.trans_dim, 3 * self.group_size, 1),
                nn.ReLU() if "scale" not in self.norm_attribute else nn.Tanh(),
            )

            self.rotation_head = nn.Sequential(
                nn.Conv1d(self.trans_dim, 4 * self.group_size, 1), nn.Tanh()
            )

        if "sh" in self.attribute:
            self.sh_head = nn.Sequential(
                nn.Conv1d(self.trans_dim, 3 * self.group_size, 1),
            )

        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, pts, vis=False, save=False, model_ids=None, **kwargs):
        # we do color change here in cuda and batch
        opacity_index = [3]
        scale_index = [4, 5, 6]
        rotation_index = [7, 8, 9, 10]
        sh_index = [11, 12, 13]

        neighborhood, center = self.group_divider(pts)

        center_pos = center[..., self.pos_feature_dim]
        x_vis, mask = self.MAE_encoder(neighborhood, center_pos)
        B, _, C = x_vis.shape  # B VIS C
        feature_dim = neighborhood.shape[-1]

        pos_emd_vis = self.decoder_pos_embed(center_pos[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center_pos[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        # retrieve the recosntruction target: use nearest in the potential neighbors
        if self.soft_knn:
            neighborhood = neighborhood[:, :, : self.group_size, :]

        B, M, C = x_rec.shape
        rebuild_points = (
            self.increase_dim(x_rec.transpose(1, 2))
            .transpose(1, 2)
            .reshape(B * M, -1, 3)
        )  # B M 1024
        gt_points = neighborhood[..., :3][mask].reshape(B * M, -1, 3)
        loss_dict = {}
        loss1 = chamfer_distance(rebuild_points, gt_points, norm=2)[0]
        loss_dict["cd"] = loss1

        if "opacity" in self.attribute:
            rebuild_density = (
                self.opacity_head(x_rec.transpose(1, 2))
                .transpose(1, 2)
                .reshape(B * M, -1, 1)
            )  # B M 1024
            gt_density = neighborhood[..., opacity_index][mask].reshape(B * M, -1, 1)
            # L1 loss for density
            loss2 = torch.nn.functional.l1_loss(rebuild_density, gt_density)
            loss_dict["density"] = loss2

        if "scale" in self.attribute and "rotation" in self.attribute:
            rebuild_scale = (
                self.scale_head(x_rec.transpose(1, 2))
                .transpose(1, 2)
                .reshape(B * M, -1, 3)
            )
            gt_scale = neighborhood[..., scale_index][mask].reshape(B * M, -1, 3)

            rebuild_rotation = (
                self.rotation_head(x_rec.transpose(1, 2))
                .transpose(1, 2)
                .reshape(B * M, -1, 4)
            )
            # normalize rotation
            rebuild_rotation[..., 0] = 1 - rebuild_rotation[..., 0]
            rebuild_rotation = rebuild_rotation / (
                torch.norm(rebuild_rotation, p=2, dim=-1, keepdim=True) + 1e-9
            )
            gt_rotation = neighborhood[..., rotation_index][mask].reshape(B * M, -1, 4)

            loss_scale = torch.nn.functional.l1_loss(rebuild_scale, gt_scale)  # * 0.01
            loss_rotation = torch.nn.functional.l1_loss(
                rebuild_rotation, gt_rotation
            )  # * 0.01 # try L1 first
            loss_dict["scale"] = loss_scale  # * 0.01
            loss_dict["rotation"] = loss_rotation  # * 0.01

        if "sh" in self.attribute:
            # print("x_rec", x_rec.shape) # ([128, 38, 384]) #  token M
            rebuild_sh = (
                self.sh_head(x_rec.transpose(1, 2))
                .transpose(1, 2)
                .reshape(B * M, -1, 3)
            )  # B M 1024
            gt_sh = neighborhood[..., sh_index][mask].reshape(B * M, -1, 3)

            loss3 = torch.nn.functional.l1_loss(rebuild_sh, gt_sh)  # * 0.01
            loss_dict["sh"] = loss3

        if save:
            # debug we choose first in batch
            rebuild_gaussians = [rebuild_points]
            if "opacity" in self.attribute:
                rebuild_gaussians.append(rebuild_density)
            else:
                rebuild_opacity = neighborhood[..., opacity_index][mask].reshape(
                    B * M, -1, 1
                )
                rebuild_gaussians.append(rebuild_opacity)

            if "scale" in self.attribute and "rotation" in self.attribute:
                rebuild_gaussians.append(rebuild_scale)
                rebuild_gaussians.append(rebuild_rotation)
            else:
                rebuild_scale = neighborhood[..., scale_index][mask].reshape(
                    B * M, -1, 3
                )
                rebuild_gaussians.append(rebuild_scale)
                rebuild_rotation = neighborhood[..., rotation_index][mask].reshape(
                    B * M, -1, 4
                )
                rebuild_gaussians.append(rebuild_rotation)

            if "sh" in self.attribute:
                rebuild_gaussians.append(rebuild_sh)

            # get back gaussian feature
            rebuild_gaussians = torch.cat(rebuild_gaussians, dim=-1)
            # print("neighborhood", neighborhood.shape)
            vis_gaussians = neighborhood[~mask].reshape(
                B * (self.num_group - M), -1, feature_dim
            )[..., :14]
            vis_gaussians[..., :3] = vis_gaussians[..., :3] + center_pos[..., :3][
                ~mask
            ].unsqueeze(
                1
            )  # xyz position back to world
            rebuild_gaussians[..., :3] = rebuild_gaussians[..., :3] + center_pos[
                ..., :3
            ][mask].unsqueeze(1)

            vis_gaussians = vis_gaussians.reshape(B, -1, vis_gaussians.shape[-1])

            rebuild_gaussians = rebuild_gaussians.reshape(
                B, -1, rebuild_gaussians.shape[-1]
            )
            full_gaussian = torch.cat([rebuild_gaussians, vis_gaussians], dim=1)
            original_gaussian = pts.clone().detach()

            return loss_dict, vis_gaussians, full_gaussian, original_gaussian
        else:
            return loss_dict


# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.soft_knn = getattr(config, "soft_knn", False)

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.attribute = config.attribute

        self.group_attribute = config.group_attribute
        self.group_divider = Group(
            num_group=self.num_group,
            group_size=self.group_size,
            attribute=config.group_attribute,
            soft_knn=self.soft_knn,
        )

        self.encoder = (
            SoftEncoder(encoder_channel=self.encoder_dims, attribute=config.attribute)
            if self.soft_knn
            else Encoder(encoder_channel=self.encoder_dims, attribute=config.attribute)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_feature_dim = []
        if "xyz" in config.group_attribute:
            self.pos_feature_dim.extend([0, 1, 2])

        if "opacity" in config.group_attribute:
            self.pos_feature_dim.append(3)

        if "scale" in config.group_attribute:
            self.pos_feature_dim.extend([4, 5, 6])

        if "rotation" in self.group_attribute:
            self.pos_feature_dim.extend([7, 8, 9, 10])

        if "sh" in config.group_attribute:
            self.pos_feature_dim.extend([11, 12, 13])

        print("self.pos_feature_dim ", self.pos_feature_dim)
        print("config.group_attribute", config.group_attribute)

        self.pos_embed = nn.Sequential(
            nn.Linear(len(self.pos_feature_dim), 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.type = config.type

        if self.type == "linear":
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, self.cls_dim)
            )
        else:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim),
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }

            for k in list(base_ckpt.keys()):
                if k.startswith("MAE_encoder"):
                    base_ckpt[k[len("MAE_encoder.") :]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith("base_model"):
                    base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )
        else:
            print_log("Training from scratch!!!", logger="Transformer")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)  # KNN
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        center_pos = center[..., self.pos_feature_dim]
        pos = self.pos_embed(center_pos)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret

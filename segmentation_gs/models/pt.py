import torch
import torch.nn as nn
import torch.nn.functional as F
import pprint
from timm.models.layers import DropPath
from logger import get_missing_parameters_message, get_unexpected_parameters_message

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from pointnet2_utils import PointNetFeaturePropagation
from models.neural_nn import N3AggregationBase


def fps_gs(data, number, attribute=['xyz'], return_idx = False):
    '''
        data B N K
        number int
    '''
    fps_index = []
    if 'xyz' in attribute:
        fps_index.extend([0,1,2])
    if 'opacity' in attribute:
        fps_index.extend([3])
    if 'scale' in attribute:
        fps_index.extend([4,5,6])
    if 'rotation' in attribute:
        fps_index.extend([7,8,9,10])
    if 'sh' in attribute:
        fps_index.extend([11,12,13])

    data_fps = data.clone()[...,fps_index].contiguous()
    # print("data_fps", data_fps.shape)
    fps_idx = pointnet2_utils.furthest_point_sample(data_fps, number) 
    if return_idx:
        return fps_idx
    # print("fps_idx", fps_idx.shape, data.transpose(1, 2).contiguous().shape)
    fps_data = pointnet2_utils.gather_operation(data.transpose(
        1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(
        1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data



class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel, attribute=['xyz']):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.attribute = attribute
        input_dim = 0
        if 'xyz' in attribute:
            input_dim += 3
        if 'opacity' in attribute:
            input_dim += 1
        if 'sh' in attribute:
            input_dim += 3
        if 'scale' in attribute:
            input_dim += 3
        if 'rotation' in attribute:
            input_dim += 4

        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3, gs B G N K
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        # choose the attribute we want
        # print("point_groups", point_groups.shape)
        attribute_index = [0, 1, 2]
        if 'opacity' in self.attribute:
            opacity_index = [3]
            attribute_index.extend(opacity_index)
        if 'scale' in self.attribute:
            scale_index = [4, 5, 6]
            attribute_index.extend(scale_index)
        if 'rotation' in self.attribute:
            rotation_index = [7,8,9,10]
            attribute_index.extend(rotation_index)
        if 'sh' in self.attribute:
            sh_index = [11,12,13]
            attribute_index.extend(sh_index)
        
        # choose the attribute we want
        # print("org point_groups", point_groups.shape)
        point_groups = point_groups[..., attribute_index]

        point_groups = point_groups.reshape(bs * g, n, -1)
        # encoder
        # print("point_groups", point_groups.shape)
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        # print("feature",  feature.shape)  ([8192, 256, 32])
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        # print("feature_global",  feature_global.shape) ([8192, 256, 1])
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)
    

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size, attribute=['xyz'], soft_knn=False):
        # attribute use to group
        # xyz is popular
        # let's try xyz + rotation

        super().__init__()
        self.num_group = num_group
        self.group_size = (
            group_size if not soft_knn else int(1.25 * group_size)
        )  # expand potential neighbor size if soft_knn
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.attribute = attribute
        attribute_index = []
        if 'xyz' in self.attribute:
            xyz_index = [0,1,2]
            attribute_index.extend(xyz_index)
        if 'opacity' in self.attribute:
            opacity_index = [3]
            attribute_index.extend(opacity_index)
        if 'scale' in self.attribute:
            scale_index = [4,5,6]
            attribute_index.extend(scale_index)
        if 'rotation' in self.attribute:
            rotation_index = [7,8,9,10]
            attribute_index.extend(rotation_index)
        if 'sh' in self.attribute:
            sh_index = [11,12,13]
            attribute_index.extend(sh_index)

        self.attribute_index = attribute_index

    def forward(self, xyz):
        '''
            input: B N 3 or B N K
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, feature_dim = xyz.shape
        if feature_dim == 3: # only xyz 
            # fps the centers out
            center = fps(xyz, self.num_group) # B G 3
            # knn to get the neighborhood
            
            _, idx = self.knn(xyz, center) # B G M
            assert idx.size(1) == self.num_group
            assert idx.size(2) == self.group_size
            idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)
            neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
            neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
            # normalize
            neighborhood = neighborhood - center.unsqueeze(2)
        else: # gaussian attribute
            center = fps_gs(xyz, self.num_group, self.attribute) # B G K
            # choose center based on xyz
            # choose neighbor based on new attribute
            center_group = center[...,self.attribute_index]
            xyz_group = xyz[...,self.attribute_index]

            _, idx = self.knn(xyz_group, center_group) # B G M
            assert idx.size(1) == self.num_group
            assert idx.size(2) == self.group_size          
            idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)
            neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
            neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, -1).contiguous()
            neighborhood[...,:3] = neighborhood[...,:3] - center.unsqueeze(2)[...,:3]

        return neighborhood, center
    


class SoftEncoder(nn.Module):
    def __init__(self, encoder_channel, k=32, attribute=['xyz'], temp_opt={}):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.attribute = attribute
        self.k = k
        # self.group_size = group_size
        # self.num_group = num_group

        input_dim = 3
        if 'opacity' in attribute:
            input_dim += 1
        if 'color' in attribute:
            input_dim += 3
        if 'sh' in attribute:
            input_dim += 3
        if 'scale' in attribute:
            input_dim += 3
        if 'rotation' in attribute:
            input_dim += 4

        # Initial convolution layers for embedding
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.encoder_channel, 1)
        )

        # NeuralNearestNeighbors and aggregation
        self.n3_agg = N3AggregationBase(k, temp_opt=temp_opt)

    def forward(self, point_groups):
        '''
        Args:
            point_groups: Tensor of shape (B, G, O, K)
        Returns:
            z: Tensor of shape (B, G, C), where C is the dimension of the encoded token feature
        '''
        bs, g, o, k = point_groups.shape
        point_groups = point_groups.reshape(bs * g, o, k).contiguous()

        attribute_index = [0, 1, 2]
        if 'opacity' in self.attribute:
            opacity_index = [3]
            attribute_index.extend(opacity_index)
        if 'scale' in self.attribute:
            scale_index = [4, 5, 6]
            attribute_index.extend(scale_index)
        if 'rotation' in self.attribute:
            rotation_index = [7, 8, 9, 10]
            attribute_index.extend(rotation_index)
        if 'sh' in self.attribute:
            sh_index = [11, 12, 13]
            attribute_index.extend(sh_index)
        if 'color' in self.attribute:  # color
            color_index = list(range(59, 62))  # 59 62
            attribute_index.extend(color_index)
        point_groups = point_groups[..., attribute_index]
        point_groups = point_groups.contiguous().reshape(bs * g, o, len(attribute_index))

        # Tokenization via first convolution layer
        feature = self.first_conv(point_groups.transpose(
            2, 1).contiguous())  # (BG, E, O)
        feature_global = torch.max(feature, dim=2, keepdim=True)[
            0]  # (BG, E, 1)

        # Preparing for N3 aggregation
        xe = feature  # (BG, E, O), use as database items
        ye = feature_global  # (BG, E, 1), use as query items

        # Perform N3 aggregation
        # (BG, E, k), k is the neighbor num in the hard knn setting
        z = self.n3_agg(xe, ye)
        # TODO, could try then use the same soft knn again to find 1 neighbor out of k, essentially aggerate to group feature

        z = torch.cat([torch.max(z, dim=2, keepdim=True)[0].repeat(
            1, 1, self.k), z], dim=1)  # append the max pooling to local feature
        z = self.second_conv(z)  # (BG, C, k)
        z_global = torch.max(z, dim=2, keepdim=False)[0]  # (BG, C)

        # output: (B, G, C)
        return z_global.reshape(bs, g, -1)

    def compute_indices(self, features):
        '''
        Compute indices for potential neighbors
        Args:
            features: Tensor of shape (B, G, 512)
        Returns:
            I: Index tensor of shape (B, G, O)
        '''
        B, G, _ = features.shape
        _, indices = torch.topk(features, self.group_size, dim=2)
        indices = indices.unsqueeze(2).expand(B, G, self.group_size, -1)
        return indices


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(
                    drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class get_model(nn.Module):
    def __init__(self, cls_dim, args=None):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.cls_dim = cls_dim
        self.num_heads = 6

        self.group_size = 32
        self.num_group = args.num_group
        # grouper
        self.group_divider = Group(
            num_group=self.num_group, 
            group_size=self.group_size, 
            attribute=args.group_attribute,
            soft_knn=args.soft_knn
            )
        # define the encoder
        self.encoder_dims = 384
        # self.encoder = Encoder(encoder_channel=self.encoder_dims, attribute=args.attribute)
        self.encoder = Encoder(encoder_channel=self.encoder_dims, attribute=args.attribute) if not args.soft_knn else \
            SoftEncoder(encoder_channel=self.encoder_dims,
                        attribute=args.attribute)

        self.pos_feature_dim = []
        if 'xyz' in args.group_attribute:
            self.pos_feature_dim.extend([0,1,2])

        if 'opacity' in args.group_attribute:
   
            self.pos_feature_dim.append(3)

        if 'scale' in args.group_attribute:
    
            self.pos_feature_dim.extend([4,5,6])

        if 'rotation' in args.group_attribute:

            self.pos_feature_dim.extend([7,8,9,10])

        if  'sh' in args.group_attribute:
       
            self.pos_feature_dim.extend([11,12,13])

        print("group_attribute",args.group_attribute)
        print("self.pos_feature_dim)", self.pos_feature_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(len(self.pos_feature_dim), 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3,
                                                        mlp=[self.trans_dim * 4, 1024])

        self.convs1 = nn.Conv1d(3392, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k,
                         v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                    get_missing_parameters_message(incompatible.missing_keys)
                )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                    get_unexpected_parameters_message(
                        incompatible.unexpected_keys)

                )

            print(
                f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, gs_data, cls_label, pc_xyz):
        # TODO color changes

        
        B, C, N = gs_data.shape
        N_pc = pc_xyz.shape[-2]
        gs_data = gs_data.transpose(-1, -2)  # B N 3
        gs_xyz = gs_data[..., :3]
        # divide the point clo  ud in the same form. This is important
        # print("gs_data", gs_data.shape)
        # print("gs_data", gs_data.shape)
        neighborhood, center = self.group_divider(gs_data)
        center_xyz = center[..., :3]
        group_input_tokens = self.encoder(neighborhood)  # B G N
        center_pos = center[...,self.pos_feature_dim]
        

        pos = self.pos_embed(center_xyz)
        # final input
        x = group_input_tokens
        # transformer
        feature_list = self.blocks(x, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous()
                        for x in feature_list]
        x = torch.cat(
            (feature_list[0], feature_list[1], feature_list[2]), dim=1)  # 1152
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat(
            (x_max_feature, x_avg_feature, cls_label_feature), 1)  # 1152*2 + 64

        f_level_0 = self.propagation_0(
            pc_xyz.transpose(-1, -2), center_xyz.transpose(-1, -2), pc_xyz.transpose(-1, -2), x)


        x = torch.cat((f_level_0, x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss

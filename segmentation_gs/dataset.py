import numpy as np
import os
import json
import math
import h5py
import numpy as np
import os

from plyfile import PlyData
import math 
from torch.utils.data import Dataset

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        # elif file_extension in ['.pcd']:
        #     return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.ply']:
            return cls._read_ply(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]
    
    @classmethod
    def _read_ply(cls, file_path):
        return PlyData.read(file_path)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_gaussian_attribute(vertex, attribute):
    # assert 'xyz' in attribute, 'At least need xyz attribute'
    if 'xyz' in attribute:
        x = vertex['x'].astype(np.float32)
        y = vertex['y'].astype(np.float32)
        z = vertex['z'].astype(np.float32)
        data = np.stack((x, y, z), axis=-1) # [n, 3]

    if 'opacity' in attribute:
        opacity = vertex['opacity'].astype(np.float32).reshape(-1, 1)
        opacity = np_sigmoid(opacity)
        # opacity range from 0 to 1
        data = np.concatenate((data, opacity), axis=-1)
        # print("data", data.shape)

    if 'scale' in attribute and 'rotation' in attribute:
        scale_names = [
            p.name
            for p in vertex.properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((data.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)
        
        scales = np.exp(scales)  # scale normalization
        
        # print("scales", scales.min(), scales.max())

        rot_names = [
            p.name for p in vertex.properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((data.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        # always set the first to be positive
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]

        data = np.concatenate((data, scales, rots), axis=-1)
 
    if 'sh' in attribute:
        # sphere homrincals to rgb
        features_dc = np.zeros((data.shape[0], 3, 1))
        features_dc[:, 0, 0] = vertex['f_dc_0'].astype(np.float32)
        features_dc[:, 1, 0] = vertex['f_dc_1'].astype(np.float32)
        features_dc[:, 2, 0] = vertex['f_dc_2'].astype(np.float32)
  
        feature_pc = features_dc.reshape(-1, 3)
        data = np.concatenate((data, feature_pc), axis=1)


    return data


class PartNormalGSDataset(Dataset):
    def __init__(self, 
                 partanno_root='./shapenetcore_partanno_root_segmentation_benchmark_v0_normal',
                 gs_root='./data/gs_data', 
                 pc_to_gs_map= './split_to_org_gs_map.json',
                 npoints=2500, 
                 split='train', 
                 attribute=['xyz','opacity','scale','rotation','sh'], 
                 norm_attribute=['xyz'],
                 class_choice=None, 
                 normal_channel=False):

        self.npoints = npoints
        self.gs_root = gs_root
        self.pc_root = os.path.join(partanno_root,'shape_data')
        self.partanno_root = partanno_root
        self.catfile = os.path.join(partanno_root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.attribute = attribute
        self.norm_attribute = norm_attribute


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                # print("ls", ls[0], ls[1])
                self.cat[ls[0]] = ls[1]
                # print("self.cat ", self.cat)
                # {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}

        with open(pc_to_gs_map, 'r') as f:
            self.pc_to_gs_map = json.load(f)

        self.cat = {k: v for k, v in self.cat.items()}

        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if class_choice is not None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.partanno_root,'train_test_split' ,'shuffled_train_file_list.json'), 'r') as f:
            split_load = json.load(f)
        train_ids = []
        for train_path in split_load:
            cat_id = train_path.split('/')[-2]
            obj_id = train_path.split('/')[-1]
            train_ids.append(f'{cat_id}-{obj_id.split(".")[0]}')

        with open(os.path.join(self.partanno_root,'train_test_split' , 'shuffled_val_file_list.json'), 'r') as f:
            split_load = json.load(f)
        val_ids = []
        for val_path in split_load:
            cat_id = val_path.split('/')[-2]
            obj_id = val_path.split('/')[-1]
            val_ids.append(f'{cat_id}-{obj_id.split(".")[0]}')

        with open(os.path.join(self.partanno_root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            split_load = json.load(f)
        test_ids = []
        for test_path in split_load:
            cat_id = test_path.split('/')[-2]
            obj_id = test_path.split('/')[-1]
            test_ids.append(f'{cat_id}-{obj_id.split(".")[0]}')

  
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []

            if split == 'trainval': 
                fns = []
                for id in train_ids:
                    if id.split('-')[0] == self.cat[item]:
                        fns.append(self.pc_to_gs_map[id])
                for id in val_ids:
                    if id.split('-')[0] == self.cat[item]:
                        fns.append(self.pc_to_gs_map[id])

            elif split == 'train':
                fns = []
                for id in train_ids:
                    if id.split('-')[0] == self.cat[item]:
                        fns.append(self.pc_to_gs_map[id])
            
            elif split == 'val':
                fns = []
                for id in val_ids:
                    if id.split('-')[0] == self.cat[item]:
                        fns.append(self.pc_to_gs_map[id])

            elif split == 'test':
                fns = []
                for id in test_ids:
                    if id.split('-')[0] == self.cat[item]:
                        fns.append(self.pc_to_gs_map[id])
            
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                cat_id = fn.split('-')[0]
                obj_id = fn.split('-')[1].split('.')[0]
                self.meta[item].append(os.path.join(self.pc_root, cat_id, obj_id + '.txt'))


        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # print("fn", fn)
                # print("item", item)
                self.datapath.append((item, fn))


        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}


        # TODO do not cache
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def pc_norm_gs(self, pc, attribute=['xyz'], extra_pc=None):
        """ pc: NxC, return NxC """
        pc_xyz = pc[..., :3]
        centroid = np.mean(pc_xyz, axis=0)
        pc_xyz = pc_xyz - centroid
        m = np.max(np.sqrt(np.sum(pc_xyz**2, axis=1)))
        pc_xyz = pc_xyz / m
        # inside a sphere
        pc[..., :3] = pc_xyz
        pc[..., 4:7] = pc[..., 4:7] / m 


        if extra_pc is not None:
            extra_pc[..., :3] = (extra_pc[..., :3] - centroid) / m

        if 'opacity' in attribute:
            # normalize to a -1 to 1 range
            min_opacity = 0
            max_opacity = 1
            pc[..., 3] = (pc[..., 3] - min_opacity) / \
                (max_opacity - min_opacity) * 2 - 1

        if 'scale' in attribute:
            # normalize to a -1 to 1 range
            s_center = np.mean(pc[..., 4:7], axis=0)
            pc[..., 4:7] = pc[..., 4:7] - s_center
            s_m = np.max(np.sqrt(np.sum(pc[..., 4:7]**2, axis=1)))
            pc[..., 4:7] = pc[..., 4:7] / s_m
        else:
            s_center = np.zeros(3)
            s_m = 1

        if 'sh'  in attribute:
            sh = pc[...,11:14]
            sh = sh * 0.28209479177387814 
            sh = np.clip(sh, -0.5, 0.5)
            sh = 2 * sh / math.sqrt(3)  
            pc[...,11:14] = sh


        if extra_pc is not None:
            return pc, extra_pc
        return pc

    def __getitem__(self, index):
        if index in self.cache:
            gs_data, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat_id = self.cat[fn[0]]
            obj_id = fn[1].split('/')[-1].split('.')[0].split('-')[-1]
            pc_path = os.path.join(self.pc_root, cat_id, obj_id + '.txt')
            pc_data = np.loadtxt(pc_path).astype(np.float32)
            pc_xyz = pc_data[..., :3]
            pc_label = pc_data[..., -1].astype(np.int32)

            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
     
            cat_id = self.cat[cat]
            gs_path = os.path.join(self.gs_root, self.pc_to_gs_map[f'{cat_id}-{obj_id}'])
            try:
                gs = IO.get(gs_path)
            except Exception as e:
                print(f"Error {e} in loading", gs_path)
        
            vertex = gs['vertex']
            gs_data = read_gaussian_attribute(vertex, self.attribute)

    
        rot_mat = np.array([[1, 0, 0], 
                            [0, 0, 1], 
                            [0, -1, 0]])
        # rotate xyz to gs coordinate
        pc_xyz = pc_xyz @ rot_mat

        gs_data,  pc_xyz  = self.pc_norm_gs(gs_data, self.attribute, extra_pc=pc_xyz)
        ###################
        # very importatnt, rotation gs data along x axis with -90 deg
        ###################

        # since gaussian number might different from point_set, run another sampling process
        choice_gs = np.random.choice(len(gs_data), self.npoints, replace=True) # self.npoints
        gs_data = gs_data[choice_gs, :]
        choice_pc = np.random.choice(len(pc_xyz), self.npoints, replace=True)
        pc_xyz = pc_xyz[choice_pc, :]
        seg = pc_label[choice_pc]

        return gs_data, cls, pc_xyz, seg

    def __len__(self):
        return len(self.datapath)

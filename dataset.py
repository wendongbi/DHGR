# my own pyg dataset class
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url
import shutil
from scipy.io import loadmat
import os
import numpy as np
from torch_geometric.nn.conv.gcn_conv import GCNConv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from dataset_benchmark import load_nc_dataset
from data_utils import load_fixed_splits

def onehot_encoder_dim(values):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoding = onehot_encoder.fit_transform(integer_encoded)
    return torch.from_numpy(onehot_encoding)

def onehot_encoder(x):
    x_onehot = None
    for col_idx in range(x.shape[1]):
        col = onehot_encoder_dim(x[:, col_idx])
        if x_onehot is None:
            x_onehot = col
        else:
            x_onehot = torch.cat([x_onehot, col], dim=1)
    return x_onehot.float()


# flag, gender, major, second major/minor (if applicable), dorm/house,
# year, and high school
class Facebook100(InMemoryDataset):
    def __init__(self, root, dataset, transform=None, pre_transform=None,
                split='random', num_val=500, num_test=None, num_train_per_class=20, to_onehot=True, train_val_test_ratio=[0.6,0.2,0.2]): # 0.4,0.3,0.3 | 0.2,0.4,0.4 | 0.6,0.2,0.2
        self.root = root
        self.dataset = dataset
        self.train_val_test_ratio = train_val_test_ratio
        super().__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        if to_onehot:
            self.data.x = onehot_encoder(self.data.x)
        if split != None:
            self.split_(split, num_val, num_test, num_train_per_class)

    @property
    def raw_file_names(self):
        # file_map = {
        #     'Amherst': 'Amherst41.mat',
        #     'Hamilton': 'Hamilton46.mat',
        #     'Georgetown': 'Georgetown15.mat',
        #     'Penn': 'Penn94.mat'

        # }
        # assert self.dataset in file_map
        # file_name = file_map[self.dataset]
        file_name = self.dataset + '.mat'
        return [file_name]

    @property
    def processed_file_names(self):
        return ['data.pt']
    def split_(self, split, num_val, num_test, num_train_per_class):
        data = self.get(0)
        lbl_num = data.y.shape[0]
        data.train_mask = torch.BoolTensor([False] * lbl_num)
        data.val_mask = torch.BoolTensor([False] * lbl_num)
        data.test_mask = torch.BoolTensor([False] * lbl_num)
        if split == 'random':
            if self.train_val_test_ratio is None:
                for c in range(self.num_classes):
                    idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                    idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                    data.train_mask[idx] = True

                remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
                remaining = remaining[torch.randperm(remaining.size(0))]

                data.val_mask[remaining[:num_val]] = True
                if num_test is not None:
                    data.test_mask[remaining[num_val:num_val + num_test]] = True
                else:
                    data.test_mask[remaining[num_val:]] = True
                self.data, self.slices = self.collate([data])
            else:
                for c in range(self.num_classes):
                    idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                    num_class = len(idx)
                    num_train_per_class = int(np.ceil(num_class * self.train_val_test_ratio[0]))
                    num_val_per_class = int(np.floor(num_class * self.train_val_test_ratio[1]))
                    num_test_per_class = num_class - num_train_per_class - num_val_per_class
                    print(num_train_per_class, num_val_per_class, num_test_per_class)
                    assert num_test_per_class >= 0
                    idx_perm = torch.randperm(idx.size(0))
                    idx_train = idx[idx_perm[:num_train_per_class]]
                    idx_val = idx[idx_perm[num_train_per_class:num_train_per_class+num_val_per_class]]
                    idx_test = idx[idx_perm[num_train_per_class+num_val_per_class:]]
                    data.train_mask[idx_train] = True
                    data.val_mask[idx_val] = True
                    data.test_mask[idx_test] = True

                self.data, self.slices = self.collate([data])
            
            
    
    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        # source_dir = '/users/wendong/data/facebook100'
        source_dir = '../data/facebook100'
        shutil.copyfile(os.path.join(source_dir, self.raw_file_names[0]), os.path.join(self.raw_dir, self.raw_file_names[0]))

    def process(self):
        # Read data into huge `Data` list.
        mat = loadmat(os.path.join(self.raw_dir, self.raw_file_names[0]))
        adj = mat['A']
        edge_index = list(adj.nonzero())
        for idx in range(len(edge_index)):
            edge_index[idx] = torch.from_numpy(edge_index[idx].astype(np.int64))
        edge_index = torch.stack(edge_index, dim=0)
        x = torch.from_numpy(mat['local_info'][:, 1:].astype(np.float32)).float()
        y = torch.from_numpy(mat['local_info'][:, 0].astype(np.int64)).long()

        data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
        data_list = [data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



def idx2mask(idx, N):
    mask = torch.zeros((idx.shape[0], N))
    for i in range(idx.shape[0]):
        mask[i][idx[i]] = 1
    return mask.bool().transpose(0, 1)

def prepare_dataset_benchmark(dataname, sub_dataname=''):
    dataset = load_nc_dataset(dataname=dataname, sub_dataname=sub_dataname)
    data = torch_geometric.data.Data(x=dataset.graph['node_feat'], edge_index=dataset.graph['edge_index'], y=dataset.label.squeeze(-1))
    split = load_fixed_splits(dataname, sub_dataset=sub_dataname)
    train_idx = torch.stack([split[i]['train'] for i in range(len(split))], dim=0)
    val_idx = torch.stack([split[i]['valid'] for i in range(len(split))], dim=0)
    test_idx = torch.stack([split[i]['test'] for i in range(len(split))], dim=0)
    print(train_idx.shape, val_idx.shape, test_idx.shape)
    data.train_mask = idx2mask(train_idx, data.num_nodes)
    data.val_mask = idx2mask(val_idx, data.num_nodes)
    data.test_mask = idx2mask(test_idx, data.num_nodes)
    return data
# data = prepare_dataset_benchmark('arxiv-year')
# print(data)


# flag, gender, major, second major/minor (if applicable), dorm/house,
# year, and high school
class Facebook100_heterphily(InMemoryDataset):
    def __init__(self, root, dataset, transform=None, pre_transform=None,
                split=None, num_val=500, num_test=None, num_train_per_class=20, to_onehot=True, train_val_test_ratio=[0.6,0.2,0.2]): # 0.4,0.3,0.3 | 0.2,0.4,0.4 | 0.6,0.2,0.2
        self.root = root
        self.dataset = dataset
        self.train_val_test_ratio = train_val_test_ratio
        super().__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        if to_onehot:
            self.data.x = onehot_encoder(self.data.x)
        if split != None:
            self.split_(split, num_val, num_test, num_train_per_class)
    @property
    def raw_file_names(self):
        # file_map = {
        #     'Amherst': 'Amherst41.mat',
        #     'Hamilton': 'Hamilton46.mat',
        #     'Georgetown': 'Georgetown15.mat',
        #     'Penn': 'Penn94.mat'

        # }
        # assert self.dataset in file_map
        # file_name = file_map[self.dataset]
        file_name = self.dataset + '.mat'
        return [file_name]

    @property
    def processed_file_names(self):
        return ['data.pt']
    def split_(self, split, num_val, num_test, num_train_per_class):
        data = self.get(0)
        lbl_num = data.y.shape[0]
        data.train_mask = torch.BoolTensor([False] * lbl_num)
        data.val_mask = torch.BoolTensor([False] * lbl_num)
        data.test_mask = torch.BoolTensor([False] * lbl_num)
        if split == 'random':
            if self.train_val_test_ratio is None:
                for c in range(self.num_classes):
                    idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                    idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                    data.train_mask[idx] = True

                remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
                remaining = remaining[torch.randperm(remaining.size(0))]

                data.val_mask[remaining[:num_val]] = True
                if num_test is not None:
                    data.test_mask[remaining[num_val:num_val + num_test]] = True
                else:
                    data.test_mask[remaining[num_val:]] = True
                self.data, self.slices = self.collate([data])
            else:
                for c in range(self.num_classes):
                    idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                    num_class = len(idx)
                    num_train_per_class = int(np.ceil(num_class * self.train_val_test_ratio[0]))
                    num_val_per_class = int(np.floor(num_class * self.train_val_test_ratio[1]))
                    num_test_per_class = num_class - num_train_per_class - num_val_per_class
                    print(num_train_per_class, num_val_per_class, num_test_per_class)
                    assert num_test_per_class >= 0
                    idx_perm = torch.randperm(idx.size(0))
                    idx_train = idx[idx_perm[:num_train_per_class]]
                    idx_val = idx[idx_perm[num_train_per_class:num_train_per_class+num_val_per_class]]
                    idx_test = idx[idx_perm[num_train_per_class+num_val_per_class:]]
                    data.train_mask[idx_train] = True
                    data.val_mask[idx_val] = True
                    data.test_mask[idx_test] = True
                self.data, self.slices = self.collate([data])
            
            
    
    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        source_dir = '/users/wendong/data/facebook100'
        shutil.copyfile(os.path.join(source_dir, self.raw_file_names[0]), os.path.join(self.raw_dir, self.raw_file_names[0]))

    def process(self):
        # Read data into huge `Data` list.
        mat = loadmat(os.path.join(self.raw_dir, self.raw_file_names[0]))
        adj = mat['A']
        edge_index = list(adj.nonzero())
        for idx in range(len(edge_index)):
            edge_index[idx] = torch.from_numpy(edge_index[idx].astype(np.int64))
        edge_index = torch.stack(edge_index, dim=0)

        # # original: use flag as label
        # x = torch.from_numpy(mat['local_info'][:, 1:].astype(np.float32)).float()
        # y = torch.from_numpy(mat['local_info'][:, 0].astype(np.int64)).long()
        # -> new: use gender as label 
        x_idxs = [0, 2, 3, 4, 5, 6]
        x = torch.from_numpy(mat['local_info'][:, x_idxs].astype(np.float32)).float()
        y = torch.from_numpy(mat['local_info'][:, 1].astype(np.int64)).long()

        data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
        data.y -= 1
        data_list = [data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
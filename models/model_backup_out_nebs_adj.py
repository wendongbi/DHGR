from builtins import NotImplementedError
from functools import reduce
import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch import Tensor
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul, set_diag
from torch_geometric.nn.conv import MessagePassing, gat_conv, gcn_conv, sage_conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import SplineConv, GATConv, GATv2Conv, SAGEConv, GCNConv, GCN2Conv, GENConv, DeepGCNLayer, APPNP, JumpingKnowledge, GINConv
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from typing import Union, Tuple, Optional
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch_sparse


class GINNet(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, hidden=16):
        super(GINNet, self).__init__()
        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(
                GINConv(Linear(dataset.num_features, dataset.num_classes))
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(GINConv(Linear(dataset.num_features, hidden), train_eps=True))
                elif num == layer_num - 1:
                    self.convs.append(GINConv(Linear(hidden, dataset.num_classes), train_eps=True))
                else:
                    self.convs.append(GINConv(Linear(hidden, hidden), train_eps=True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return F.log_softmax(x, dim=1)


class JKNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, mode='cat'):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        # self.bns = torch.nn.ModuleList()
        # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.jump = JumpingKnowledge(mode=mode, channels=hidden_channels, num_layers=num_layers)
        if mode == 'cat':
            self.lin = Linear(num_layers * hidden_channels, out_channels)
        else:
            self.lin = Linear(hidden_channels, out_channels)

        self.dropout = dropout
        self.adj_t_cache = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()

        self.jump.reset_parameters()
        self.lin.reset_parameters()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.adj_t_cache is None:
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            self.adj_t_cache = gcn_norm(self.adj_t_cache)
            # self.adj_t_cache = adj_norm(self.adj_t_cache, norm='symmetric')
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, self.adj_t_cache)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]

        x = self.jump(xs)
        x = self.lin(x)

        return F.log_softmax(x, dim=-1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, hidden=16):
        super().__init__()
        self.lin1 = Linear(dataset.num_features, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.prop1 = APPNP(10, 0.1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

class DeeperGCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = Linear(dataset.num_features, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.node_encoder(x)

        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

class GCN2(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))
        self.adj_t_cache = None
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.adj_t_cache is None:
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            self.adj_t_cache = gcn_norm(self.adj_t_cache)
            # self.adj_t_cache = adj_norm(self.adj_t_cache, norm='symmetric')
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, self.adj_t_cache)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return F.log_softmax(x, dim=1)

class ConvNet(torch.nn.Module):
    def __init__(self, dataset):
        super(ConvNet, self).__init__()
        self.conv1 = SplineConv(dataset.num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, dataset.num_classes, dim=1, kernel_size=2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(in_size, hidden)
        # self.hidden_layer = nn.Linear(hidden, hidden)
        self.out_layer = nn.Linear(hidden, out_size)
        self.reset_parameters()
        # self.adj = torch.sparse_coo_tensor([np.arange(0, 4040), np.arange(0, 4040)], torch.ones(4040), size=(4040, 4040)).requires_grad_(False)
        # self.adj = torch.eye(4040).to_sparse().requires_grad_(False)
    def reset_parameters(self):
        self.input_layer.reset_parameters()
        # self.hidden_layer.reset_parameters()
        self.out_layer.reset_parameters()
    def forward(self, data):
        x = data.x
        x = self.input_layer(x)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)
        x = self.out_layer(x)
        logits = F.log_softmax(x, dim=1)
        return logits
    def get_emb(self, data):
        x = data.x
        x = self.input_layer(x)
        x = F.relu(x)
        return x
    def get_logits(self, data):
        x = data.x
        x = self.input_layer(x)
        x = F.relu(F.dropout(x, p=0.5, training=self.training))
        logits = self.out_layer(x)
        return logits

class GCNNet(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, hidden=16):
        super(GCNNet, self).__init__()
        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(
                GCNConv(dataset.num_features, dataset.num_classes)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(GCNConv(dataset.num_features, hidden))
                elif num == layer_num - 1:
                    self.convs.append(GCNConv(hidden, dataset.num_classes))
                else:
                    self.convs.append(GCNConv(hidden, hidden))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_emb(self, data):
        x, edge_index = data.x, data.edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                # x = conv(x, edge_index)
                continue
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return x
    def get_logits(self, data):
        x, edge_index = data.x, data.edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
                # continue
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return x

class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads,
                 dropout, att_dropout):
        super(GATv2, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=att_dropout, concat=True))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, dropout=att_dropout, concat=False))

        self.dropout = dropout
        self.adj_t_cache = None


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        assert isinstance(edge_index, torch.Tensor)
        if  self.adj_t_cache == None:
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            # self.adj_t_cache = adj_norm(self.adj_t_cache, norm='row')

        # add self loop
        if isinstance(edge_index, Tensor):
            num_nodes = x.size(0)
            edge_index, edge_attr = remove_self_loops(
                edge_index)
            edge_index, edge_attr = add_self_loops(
                edge_index, fill_value='mean',
                num_nodes=num_nodes)
        elif isinstance(edge_index, SparseTensor):
            if self.edge_dim is None:
                edge_index = set_diag(edge_index)
            else:
                raise NotImplementedError(
                    "The usage of 'edge_attr' and 'add_self_loops' "
                    "simultaneously is currently not yet supported for "
                    "'edge_index' in a 'SparseTensor' form")

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index) # for GATv2Conv
            # x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index) # for GATv2Conv
        
        return x.log_softmax(dim=-1)


class GAT(torch.nn.Module):
    def __init__(self, dataset, hidden=16, head=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            hidden,
            heads=head,
            concat=True,
            dropout=0.6)
        self.conv2 = GATConv(
            hidden * head,
            dataset.num_classes,
            heads=1,
            concat=False,
            dropout=0.6)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_emb(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, hidden=16, root_weight=True):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(
                SAGEConv(dataset.num_features, dataset.num_classes, root_weight=root_weight)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(SAGEConv(dataset.num_features, hidden, root_weight=root_weight))
                elif num == layer_num - 1:
                    self.convs.append(SAGEConv(hidden, dataset.num_classes, root_weight=root_weight))
                else:
                    self.convs.append(SAGEConv(hidden, hidden, root_weight=root_weight))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # adj_sp = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]).to(x.device), sparse_sizes=(x.shape[0], x.shape[0]))
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)


        # x, edge_index = data.x, data.edge_index
        # adj_sp = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]).to(x.device), sparse_sizes=(x.shape[0], x.shape[0]))
        # for ind, conv in enumerate(self.convs):
        #     if ind == len(self.convs) -1:
        #         x = conv(x, adj_sp)
        #     else:
        #         x = F.relu(conv(x, adj_sp))
        #         x = F.dropout(x, p=0.5, training=self.training)
        #         # x = conv(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_emb(self, data, layer_num=1):
        x, edge_index = data.x, data.edge_index
        adj_sp = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]).to(x.device), sparse_sizes=(x.shape[0], x.shape[0]))
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                # x = conv(x, adj_sp)
                continue
            else:
                x = F.relu(conv(x, adj_sp))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return x
    
    def get_logits(self, data, layer_num=1):
        x, edge_index = data.x, data.edge_index
        adj_sp = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]).to(x.device), sparse_sizes=(x.shape[0], x.shape[0]))
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, adj_sp)
            else:
                x = F.relu(conv(x, adj_sp))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return x


class MMD_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, device):
        super().__init__(aggr='add',)  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.device = device

    def gaussian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        """计算Gram核矩阵
        source: sample_size_1 * feature_size 的数据
        target: sample_size_2 * feature_size 的数据
        kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
        kernel_num: 表示的是多核的数量
        fix_sigma: 表示是否使用固定的标准差
            return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                            矩阵，表达形式:
                            [   K_ss K_st
                                K_ts K_tt ]
        """
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0) # 合并在一起

        total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                        int(total.size(0)), \
                                        int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                        int(total.size(0)), \
                                        int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|
        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                    bandwidth_temp in bandwidth_list]
        ret = sum(kernel_val)
        if self.check_nan(ret):
            print('ret nan!', self.check_nan(kernel_val[0]), bandwidth_list)
        return ret # 将多个核合并在一起

    def check_nan(self, value):
        return torch.any(torch.isnan(value))

    def mmd(self, source, target, kernel_mul=2.0, kernel_num=1, fix_sigma=0.1):
        n = int(source.size()[0])
        m = int(target.size()[0])

        kernels = self.gaussian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n] 
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
        XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

        YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
        YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target
        ret = (XX + XY).sum() + (YX + YY).sum()
        # ret[torch.isnan(ret)] = -1
        return ret

    def get_mmd(self, x, edge_index):
        # print('x requires grad', x.requires_grad)
        enum = edge_index.shape[1]
        adj_t = torch.sparse.LongTensor(edge_index, torch.ones(enum).to(self.device), torch.Size((x.shape[0], x.shape[0]))).t()
        row, col = edge_index
        mat_mmd = torch.zeros(enum).to(self.device)
        for idx in range(enum):
            j, i = row[idx], col[idx]
            neighbors_i = adj_t[i].coalesce().indices().view(-1)
            neighbors_j = adj_t[j].coalesce().indices().view(-1)
            mmd = self.mmd(x[neighbors_j], x[neighbors_i])
            mat_mmd[idx] = mmd
        # mat_mmd[torch.isnan(mat_mmd)] = -1
        # print('nan_num:', sum(mat_mmd == -1).item())
        # print(edge_index)
        return mat_mmd

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Compute mmd
        mat_mmd = self.get_mmd(x, edge_index)
        # print(mat_mmd.shape, norm.shape, mat_mmd)

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class MMD_GCNNet(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, device='cuda'):
        super(MMD_GCNNet, self).__init__()
        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(
                MMD_GCNConv(dataset.num_features, dataset.num_classes, device)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(MMD_GCNConv(dataset.num_features, 16, device))
                elif num == layer_num - 1:
                    self.convs.append(MMD_GCNConv(16, dataset.num_classes, device))
                else:
                    self.convs.append(MMD_GCNConv(16, 16, device))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_emb(self, data, layer_num=1):
        x, edge_index = data.x, data.edge_index
        for ind, conv in zip(range(layer_num), self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
            else:
                # x = conv(x, edge_index)
                x = F.relu(conv(x, edge_index))
                x = conv(x, edge_index)
                # x = F.dropout(x, p=0.5, training=self.training)
        return x





class MMD_SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, 
                 device: torch.device, 
                 normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(MMD_SAGEConv, self).__init__(**kwargs)
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
    
    def gaussian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        """计算Gram核矩阵
        source: sample_size_1 * feature_size 的数据
        target: sample_size_2 * feature_size 的数据
        kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
        kernel_num: 表示的是多核的数量
        fix_sigma: 表示是否使用固定的标准差
            return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                            矩阵，表达形式:
                            [   K_ss K_st
                                K_ts K_tt ]
        """
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0) # 合并在一起

        total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                        int(total.size(0)), \
                                        int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                        int(total.size(0)), \
                                        int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                    bandwidth_temp in bandwidth_list]
        ret = sum(kernel_val)
        if self.check_nan(ret):
            print('ret nan!', self.check_nan(kernel_val[0]), bandwidth_list)
        return ret # 将多个核合并在一起

    def mmd(self, source, target, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
        n = int(source.size()[0])
        m = int(target.size()[0])

        kernels = self.gaussian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n] 
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
        XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

        YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
        YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target
        ret = (XX + XY).sum() + (YX + YY).sum()
        ret[torch.isnan(ret)] = -1
        return ret

    def get_mmd(self, x, edge_index):
        enum = edge_index.shape[1]
        adj_t = torch.sparse.LongTensor(edge_index, torch.ones(enum).to(self.device), torch.Size((x.shape[0], x.shape[0]))).t()
        row, col = edge_index
        mat_mmd = torch.zeros(enum).to(self.device)
        for idx in range(enum):
            j, i = row[idx], col[idx]
            neighbors_i = adj_t[i].coalesce().indices().view(-1)
            neighbors_j = adj_t[j].coalesce().indices().view(-1)
            mmd = self.mmd(x[neighbors_j], x[neighbors_i]).detach()
            mat_mmd[idx] = mmd
        return mat_mmd

    def check_nan(self, value):
        return torch.any(torch.isnan(value))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Compute mmd
        mat_mmd = self.get_mmd(x[0], edge_index)
        # e = torch.sigmoid((mat_mmd * 10).pow(-1)) # scheme 1
        e = mat_mmd.pow(-1) # scheme 2
        # print(e.shape, e.mean().item(), e.min().item(), e.max().item(),  e.data)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, weight=e)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    # scheme 1
    # def message(self, x_j: Tensor, weight: Tensor) -> Tensor:
    #     return weight.view(-1, 1) *  x_j

    # scheme 2
    def message(self, x_j: Tensor, weight: Tensor,
                index: Tensor, ptr: OptTensor, 
                size_i: Optional[int]) -> Tensor:
        # print(weight.shape, weight.mean().item(), weight.min().item(), weight.max().item(), weight)
        gamma = softmax(weight.view(-1, 1), index, ptr, size_i)
        # print(gamma.shape, gamma.mean().item(), gamma.min().item(), gamma.max().item(), gamma.view(-1))
        return gamma *  x_j
    # def message(self, x_j: Tensor) -> Tensor:
    #     return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MMD_SAGENet(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, device=torch.device('cuda')):
        super(MMD_SAGENet, self).__init__()
        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(
                MMD_SAGEConv(dataset.num_features, dataset.num_classes, device)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(MMD_SAGEConv(dataset.num_features, 16, device))
                elif num == layer_num - 1:
                    self.convs.append(MMD_SAGEConv(16, dataset.num_classes, device))
                    
                else:
                    self.convs.append(MMD_SAGEConv(16, 16, device))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
            else:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                # x = conv(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_emb(self, data, layer_num=1):
        x, edge_index = data.x, data.edge_index
        for ind, conv in zip(range(layer_num), self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, edge_index)
            else:
                # x = conv(x, edge_index)
                x = F.relu(conv(x, edge_index))
                x = conv(x, edge_index)
                # x = F.dropout(x, p=0.5, training=self.training)
        return x

# def adj_norm(adj, norm='row'):
#     if not adj.has_value():
#         adj = adj.fill_value(1., dtype=None)
#     # add self loop
#     adj = fill_diag(adj, 1.)
#     deg = sparsesum(adj, dim=1)
#     if norm == 'symmetric':
#         deg_inv_sqrt = deg.pow_(-0.5)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
#         adj = mul(adj, deg_inv_sqrt.view(-1, 1)) # row normalization
#         adj = mul(adj, deg_inv_sqrt.view(1, -1)) # col normalization
#     elif norm == 'row':
#         deg_inv_sqrt = deg.pow_(-1)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
#         adj = mul(adj, deg_inv_sqrt.view(-1, 1)) # row normalization
#     else:
#         raise NotImplementedError('Not implete adj norm: {}'.format(norm))
#     return adj


def adj_norm(adj, norm='row'):
    if not adj.has_value():
        adj = adj.fill_value(1., dtype=None)
    # add self loop
    adj = fill_diag(adj, 0.)
    adj = fill_diag(adj, 1.)
    deg = sparsesum(adj, dim=1)
    if norm == 'symmetric':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj = mul(adj, deg_inv_sqrt.view(-1, 1)) # row normalization
        adj = mul(adj, deg_inv_sqrt.view(1, -1)) # col normalization
    elif norm == 'row':
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj = mul(adj, deg_inv_sqrt.view(-1, 1)) # row normalization
    else:
        raise NotImplementedError('Not implete adj norm: {}'.format(norm))
    return adj

class my_SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True,
                 moment=1, 
                 mode = 'sum', #  cat or sum
                 mixhop = True, 
                 moment_att_dim = 16,
                 use_adj_norm = False,
                 N = None, # num of samples
                 device = None, 
                 use_center_moment = None,
                 use_norm = False,
                 **kwargs):  # yapf: disable
        if use_adj_norm:
            kwargs.setdefault('aggr', 'add')
        else:
            kwargs.setdefault('aggr', 'mean')
        super(my_SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.moment = moment
        self.mode = mode
        self.use_norm = use_norm
        self.mixhop = mixhop
        self.moment_att_dim = moment_att_dim
        self.use_adj_norm = use_adj_norm
        self.use_center_moment = use_center_moment
        assert use_center_moment is not None
        
        if use_norm:
            # share norm
            self.norm = nn.LayerNorm(out_channels, elementwise_affine=True) 
            # individual norm
            self.bns = torch.nn.ModuleList()
            for _ in range(moment + 1):
                # self.bns.append(torch.nn.BatchNorm1d(out_channels))
                self.bns.append(nn.LayerNorm(out_channels, elementwise_affine=True))
        print('mode:', mode)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        if mode == 'cat':
            # scheme 1
            self.lin_l = Linear(in_channels[0] * moment, out_channels, bias=bias) # moment * d -> out_dim, input is the concat feat of all moments
        elif mode == 'sum' or mode == 'mean': 
            # scheme 2
            self.lin_moment_list = nn.ModuleList()
            for _ in range(moment):
                self.lin_moment_list.append(Linear(in_channels[0], out_channels, bias=bias))
            print(moment, len(self.lin_moment_list))
            if mixhop:
                self.lin_moment_list_2hop = nn.ModuleList()
                for _ in range(moment):
                    self.lin_moment_list_2hop.append(Linear(in_channels[0], out_channels, bias=bias))
        elif mode == 'rank': 
            self.rank_beta = torch.zeros(self.moment).float().to(device)
            for m in range(1, 1+  self.moment):
                self.rank_beta[m-1] = 1 / m
            self.rank_beta = F.softmax(self.rank_beta).view(-1, 1, 1) # m * 1 * 1
            print('rank coefficients:', self.rank_beta.view(-1))
            self.lin_moment_list = nn.ModuleList()
            for _ in range(moment):
                self.lin_moment_list.append(Linear(in_channels[0], out_channels, bias=bias))
            print(moment, len(self.lin_moment_list))
            if mixhop:
                self.lin_moment_list_2hop = nn.ModuleList()
                for _ in range(moment):
                    self.lin_moment_list_2hop.append(Linear(in_channels[0], out_channels, bias=bias))
        elif mode in ['attention', 'attention-feat']:
            # scheme 2
            self.lin_self_out = Linear(in_channels[0], out_channels, bias=bias)
            self.lin_self_query = Linear(out_channels, moment_att_dim, bias=False)

            self.lin_key = Linear(out_channels, moment_att_dim, bias=False)
            # self.lin_key_self = Linear(out_channels, moment_att_dim, bias=False)
            # self.lin_key_list = nn.ModuleList()

            self.lin_moment_list = nn.ModuleList()
            for _ in range(moment):
                self.lin_moment_list.append(Linear(in_channels[0], out_channels, bias=bias))

                # self.lin_key_list.append(Linear(out_channels, moment_att_dim, bias=False))

            print(moment, len(self.lin_moment_list))
            if mode == 'attention':
                self.w_att = nn.Parameter(torch.FloatTensor(2 * moment_att_dim, 1))
            elif mode == 'attention-feat':
                self.w_att = nn.Parameter(torch.FloatTensor(2 * moment_att_dim, out_channels))
        elif mode in ['transformer']:
            # scheme 2
            self.lin_self_out = Linear(in_channels[0], out_channels, bias=bias)
            self.lin_self_query = Linear(out_channels, moment_att_dim, bias=False)
            self.lin_key_self = Linear(out_channels, moment_att_dim, bias=False)
            self.lin_key_list = nn.ModuleList()
            self.lin_query_list = nn.ModuleList()
            self.lin_moment_list = nn.ModuleList()
            for _ in range(moment):
                self.lin_moment_list.append(Linear(in_channels[0], out_channels, bias=bias))
                self.lin_key_list.append(Linear(out_channels, moment_att_dim, bias=False))
                self.lin_query_list.append(Linear(out_channels, moment_att_dim, bias=False))
            print(moment, len(self.lin_moment_list))

        elif mode == 'attn-1':
            init_attn = torch.FloatTensor(self.moment, N, out_channels).fill_(1)
            # init_attn = torch.FloatTensor(self.moment, 1, out_channels).fill_(1)
            # init_attn[1:, :, :] = 0
            # ranking init
            self.rank_beta = torch.zeros(self.moment).float().to(device)
            for m in range(1, 1+  self.moment):
                self.rank_beta[m-1] = 1 / m
            self.rank_beta = F.softmax(self.rank_beta).view(-1)
            assert len(self.rank_beta) == self.moment
            for m in range(self.moment):
                init_attn[m, :, :] = self.rank_beta[m]

            self.attn = nn.Parameter(init_attn) # (m, N, d)
            # scheme 2
            self.lin_moment_list = nn.ModuleList()
            for _ in range(moment):
                self.lin_moment_list.append(Linear(in_channels[0], out_channels, bias=bias))
            # scheme 2
            if mixhop:
                self.lin_moment_list_2hop = nn.ModuleList()
                for _ in range(moment):
                    self.lin_moment_list_2hop.append(Linear(in_channels[0], out_channels, bias=bias))
        elif mode == 'attn-2':
            num_kernel = 3 # multi-head attention
            init_attn = torch.FloatTensor(self.moment, 1, out_channels, num_kernel).fill_(1) # (m, 1, d, k)
            # init_attn[1, :, :, :] = 2
            # init_attn[1:, :, :, 1] = 2
            # init_attn[1:, :, :, 2] = 0
            # ranking init
            # self.rank_beta = torch.zeros(self.moment).float().to(device)
            # for m in range(1, 1+  self.moment):
            #     self.rank_beta[m-1] = 1 / m
            # self.rank_beta = F.softmax(self.rank_beta).view(-1)
            # assert len(self.rank_beta) == self.moment
            # for m in range(self.moment):
            #     init_attn[m, :, :] = self.rank_beta[m]

            self.attn = nn.Parameter(init_attn) # (m, N, d)
            # scheme 2
            self.lin_moment_list = nn.ModuleList()
            for _ in range(moment):
                self.lin_moment_list.append(Linear(in_channels[0], out_channels, bias=bias))
            # scheme 2
            if mixhop:
                self.lin_moment_list_2hop = nn.ModuleList()
                for _ in range(moment):
                    self.lin_moment_list_2hop.append(Linear(in_channels[0], out_channels, bias=bias))
        else:
            raise NotImplementedError

        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_norm:
            self.norm.reset_parameters()
            for norm in self.bns:
                norm.reset_parameters()
        if self.mode == 'cat':
            self.lin_l.reset_parameters()
        elif self.mode in ['sum', 'mean', 'rank']:
            for fc in self.lin_moment_list:
                fc.reset_parameters()
                # nn.init.xavier_uniform_(fc.weight, gain=1.414)
            if self.mixhop:
                for fc in self.lin_moment_list_2hop:
                    fc.reset_parameters()
                    # nn.init.xavier_uniform_(fc.weight, gain=1.414)
        elif self.mode in ['attention', 'attention-feat']:
            self.lin_self_out.reset_parameters()
            self.lin_self_query.reset_parameters()
            nn.init.xavier_uniform_(self.w_att.data, gain=1.414)

            for fc in self.lin_moment_list:
                fc.reset_parameters()

            self.lin_key.reset_parameters()
            # self.lin_key_self.reset_parameters()
            # for fc in self.lin_key_list:
            #     fc.reset_parameters()
        elif self.mode == 'transformer':
            self.lin_self_out.reset_parameters()
            self.lin_self_query.reset_parameters()
            for fc in self.lin_moment_list:
                fc.reset_parameters()
            self.lin_key_self.reset_parameters()
            for fc in self.lin_key_list:
                fc.reset_parameters()
            for fc in self.lin_query_list:
                fc.reset_parameters()
        elif self.mode in ['attn-1', 'attn-2']:
            # nn.init.xavier_uniform_(self.attn, gain=1.414)
            for fc in self.lin_moment_list:
                fc.reset_parameters()
            if self.mixhop:
                for fc in self.lin_moment_list_2hop:
                    fc.reset_parameters()
        else:
            raise NotImplementedError
    
        if self.root_weight:
            self.lin_r.reset_parameters()

    def get_attention_layer(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_row_col:Adj, size: Size = None):
        if isinstance(x, Tensor):
            # x: OptPairTensor = (x, x)
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out_list, out_list_2hop = self.propagate(edge_index, x=x, size=size, moment=self.moment)
        
        if self.mode in  ['attention', 'attention-feat']:
            h_list = []
            k_list = []
            h_self = self.lin_self_out(x[0])
            if self.use_norm:
                # h_self = F.normalize(h_self, p=2, dim=-1)
                h_self = self.norm(h_self) # ln
                # h_self = self.bns[0](h_self) # bn
                # h_self = h_self / h_self.norm(p=2, dim=-1).unsqueeze(-1) # l2
            q = self.lin_self_query(h_self).repeat(self.moment + 1, 1) # N * (m+1), D
            # k0 = self.lin_key_self(h_self)
            k0 = self.lin_key(h_self)
            h_list.append(h_self)
            # k_list.append(k0 / k0.norm(p=2, dim=-1).unsqueeze(-1))
            k_list.append(k0)
            # output for each moment of 1st-neighbors
            for idx, fc in enumerate(self.lin_moment_list):
                h = fc(out_list[idx])
                if self.use_norm:
                    # h = F.normalize(h, p=2, dim=-1)
                    h = self.norm(h) # ln
                    # h = self.bns[idx+1](h) # bn
                    # h / h.norm(p=2, dim=-1).unsqueeze(-1) # l2
                # h = self.bns[idx+1](h) # bn
                h_list.append(h)
                k = self.lin_key(h_list[-1])
                k_list.append(k)
                # k_list.append(k / k.norm(p=2, dim=-1).unsqueeze(-1))
                # k_list.append(self.lin_key_list[idx](h_list[-1]))
            attn_input = torch.cat([torch.cat(k_list, dim=0), q], dim=1)
            attn_input = F.dropout(attn_input, 0.5, training=self.training)
            e = F.elu(torch.matmul(attn_input, self.w_att)) # N*(m+1), 1
            if self.mode == 'attention':
                attention = F.softmax(e.view(len(k_list), -1).transpose(0, 1), dim=1) # N, m+1
            elif self.mode == 'attention-feat':
                attention = F.softmax(e.view(len(k_list), -1, self.out_channels).transpose(0, 1), dim=1) # N, m+1, D
        else:
            return None
            # raise NotImplementedError('Not supported attention output for mode:{}'.format(self.mode))
        return attention

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_row_col:Adj, thres_deg=0,
                size: Size = None, get_moment=False) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            # x: OptPairTensor = (x, x)
            x = (x, x)

        # if self.use_adj_norm:
        #     # edge_index = gcn_norm(edge_index, edge_weight=None, num_nodes=x[0].shape[0], improved=False, add_self_loops=True, dtype=None)
        #     edge_index = adj_norm(edge_index, norm='row')

        # propagate_type: (x: OptPairTensor)
        out_list, out_list_2hop = self.propagate(edge_index, x=x, size=size, moment=self.moment)
        
        # fusion the output from different moment and different hop
        if self.mode == 'cat':
            # scheme 1
            out = torch.cat(out_list, dim=1)
            out = self.lin_l(out)
        elif self.mode == 'sum':
            # scheme 2
            out = None
            for idx, fc in enumerate(self.lin_moment_list):
                if out == None:
                    out = fc(out_list[idx])
                else:
                    out += fc(out_list[idx])
            if thres_deg > 0:
                out_1st = fc(out_list[0])
                deg = degree(edge_row_col[1], x[0].shape[0], dtype=x[0].dtype)
                mask_deg = deg < thres_deg
                # print('[Degree]', sum(mask_deg).item(), len(mask_deg) - sum(mask_deg).item())
                # idx = torch.where(mask_deg == True)[0][0]
                # print('ori', out[idx] == out_1st[idx], idx, len(torch.where(mask_deg == True)[0]))
                out[mask_deg, :] = out_1st[mask_deg, :]
                # print('new', out[idx] == out_1st[idx])
        elif self.mode == 'mean':
            # scheme 2
            out = None
            for idx, fc in enumerate(self.lin_moment_list):
                if out == None:
                    out = fc(out_list[idx])
                else:
                    out += fc(out_list[idx])
            out /= self.moment

            if self.mixhop:
                out_2hop = None
                for idx, fc in enumerate(self.lin_moment_list_2hop):
                    if out_2hop == None:
                        out_2hop = fc(out_list_2hop[idx])
                    else:
                        out_2hop += fc(out_list_2hop[idx])
                out_2hop /= self.moment

                out = (out + out_2hop) / 2

            if thres_deg > 0:  
                out_1st = fc(out_list[0])
                deg = degree(edge_row_col[1], x[0].shape[0], dtype=x[0].dtype)
                mask_deg = deg < thres_deg
                # # debugging block
                # print('[Degree]', sum(mask_deg).item(), len(mask_deg) - sum(mask_deg).item())
                # idx = torch.where(mask_deg == True)[0][0]
                # print('ori', out[idx] == out_1st[idx], idx, len(torch.where(mask_deg == True)[0]))

                # replace the embedding of nodes with degree lower than thres_deg by 1st moment output
                out[mask_deg, :] = out_1st[mask_deg, :]
                # out[mask_deg, :] = 0

                # print('new', out[idx] == out_1st[idx])
        elif self.mode == 'rank':
            out = []
            for idx, fc in enumerate(self.lin_moment_list):
                out.append(fc(out_list[idx]))
            out = torch.stack(out, dim=0).mul(self.rank_beta).sum(0) # (m, N, d) -> (N, d)

        elif self.mode in  ['attention', 'attention-feat']:
            h_list = []
            k_list = []
            h_self = self.lin_self_out(x[0])
            if self.use_norm:
                # h_self = F.normalize(h_self, p=2, dim=-1) # l2
                h_self = self.norm(h_self) # ln
                # h_self = self.bns[0](h_self) # bn
                # h_self = h_self / h_self.norm(p=2, dim=-1).unsqueeze(-1)
            q = self.lin_self_query(h_self).repeat(self.moment + 1, 1) # N * (m+1), D
            # k0 = self.lin_key_self(h_self)
            k0 = self.lin_key(h_self)
            h_list.append(h_self)
            # k_list.append(k0 / k0.norm(p=2, dim=-1).unsqueeze(-1))
            k_list.append(k0)
            # output for each moment of 1st-neighbors
            for idx, fc in enumerate(self.lin_moment_list):
                h = fc(out_list[idx])
                if self.use_norm:
                    # h = F.normalize(h, p=2, dim=-1) # l2
                    h = self.norm(h) # ln
                    # h = self.bns[idx+1](h) # bn
                    # h = h / h.norm(p=2, dim=-1).unsqueeze(-1)
                h_list.append(h)
                k = self.lin_key(h_list[-1])
                k_list.append(k)
                # k_list.append(k / k.norm(p=2, dim=-1).unsqueeze(-1))
                # k_list.append(self.lin_key(h_list[-1] / h_list[-1].mean(-1).unsqueeze(-1)))
                # k_list.append(self.lin_key_list[idx](h_list[-1]))

            attn_input = torch.cat([torch.cat(k_list, dim=0), q], dim=1)
            attn_input = F.dropout(attn_input, 0.5, training=self.training)
            e = F.elu(torch.matmul(attn_input, self.w_att)) # N*(m+1), 1
            if self.mode == 'attention':
                attention = F.softmax(e.view(len(k_list), -1).transpose(0, 1), dim=1) # N, m+1
                out = torch.stack(h_list, dim=1).mul(attention.unsqueeze(-1)).sum(1) # N, D
            elif self.mode == 'attention-feat':
                attention = F.softmax(e.view(len(k_list), -1, self.out_channels).transpose(0, 1), dim=1) # N, m+1, D
                out = torch.stack(h_list, dim=1).mul(attention).sum(1) # N, D
            # if not self.training:
            #     for idx in range(len(k_list)):
            #         print(idx, round(k_list[idx].mean(0).mean(0).item(), 3))
            #     print('attention-mean:', attention.detach().mean(0).mean(-1).cpu().tolist())
            #     print('attention-min:', attention.detach().mean(-1).min(0)[0].cpu().tolist())
            #     print('attention-max:',  attention.detach().mean(-1).max(0)[0].cpu().tolist())
        elif self.mode == 'transformer':
            h_list = []
            k_list = []
            q_list = [] # (m+1), N, D
            h_self = self.lin_self_out(x[0])
            q0 = self.lin_self_query(h_self) 
            k0 = self.lin_key_self(h_self)
            h_list.append(h_self)
            k_list.append(k0)
            q_list.append(q0)
            # output for each moment of 1st-neighbors
            for idx, fc in enumerate(self.lin_moment_list):
                h_list.append(fc(out_list[idx]))
                k_list.append(self.lin_key_list[idx](h_list[-1]))
                q_list.append(self.lin_query_list[idx](h_list[-1]))
            mat_q = torch.stack(q_list, dim=1) # (N, m+1, D)
            mat_k = torch.stack(k_list, dim=1) # (N, m+1, D)
            mat_attn = F.elu(torch.bmm(mat_q, mat_k.transpose(1, 2))) # (N, m+1, m+1)
            attention = F.softmax(mat_attn.mean(-1), dim=-1) # (N, m+1)
            out = torch.stack(h_list, dim=1).mul(attention.unsqueeze(-1)).sum(1) # N, D
        elif self.mode in ['attn-1', 'attn-2']:
            # scheme 2
            out = []
            for idx, fc in enumerate(self.lin_moment_list):
                out.append(fc(out_list[idx]))
                # out_list[idx] = fc(out_list[idx])
            attn_score = F.softmax(self.attn, dim=0) # (m, 1, d, K)
            out = torch.stack(out, dim=0).unsqueeze(-1).mul(attn_score).sum(0).mean(-1) # (m, N, d, K) -> (N, d)
            if self.mixhop:
                out_2hop = None
                for idx, fc in enumerate(self.lin_moment_list_2hop):
                    if out_2hop == None:
                        out_2hop = fc(out_list_2hop[idx])
                    else:
                        out_2hop += fc(out_list_2hop[idx])
                out_2hop /= self.moment

                out = (out + out_2hop) / 2

            if thres_deg > 0:  
                out_1st = fc(out_list[0])
                deg = degree(edge_row_col[1], x[0].shape[0], dtype=x[0].dtype)
                mask_deg = deg < thres_deg
                out[mask_deg, :] = out_1st[mask_deg, :]
        else:
            raise NotImplementedError


        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)
            

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, out_list, out_list_2hop
        # return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor,
                              moment: int) -> Tensor:

        # adj_t = adj_t.set_value(None, layout=None)
        mu = matmul(adj_t, x[0], reduce=self.aggr)
        
        # sigma = matmul(adj_t, (x[0] - mu).pow(4), reduce=self.aggr)
        out_1hop = [mu]
        if self.mixhop:
            mu_2hop = matmul(adj_t, mu, reduce=self.aggr)
            out_2hop = [mu_2hop]
        
        if moment > 1:
            if self.use_center_moment:
                sigma = matmul(adj_t, (x[0] - mu).pow(2), reduce=self.aggr)
            else:
                sigma = matmul(adj_t, (x[0]).pow(2), reduce=self.aggr)
            
            sigma[sigma == 0] = 1e-16
            sigma = sigma.sqrt()
            out_1hop.append(sigma)
            if self.mixhop:
                sigma_2hop = matmul(adj_t, (x[0] - mu_2hop).pow(2), reduce=self.aggr)
                sigma_2hop = matmul(adj_t, sigma_2hop, reduce=self.aggr)
                sigma_2hop[sigma_2hop == 0] = 1e-16
                sigma_2hop = sigma_2hop.sqrt()
                out_2hop.append(sigma_2hop)


            for order in range(3, moment+1):
                # gamma = matmul(adj_t, (x[0] - mu).div(sigma).pow(order), reduce=self.aggr)
                # gamma = matmul(adj_t, (x[0] - mu).pow(order), reduce=self.aggr)
                gamma = matmul(adj_t, x[0].pow(order), reduce=self.aggr)
                mask_neg = None
                if torch.any(gamma == 0):
                    gamma[gamma == 0] = 1e-16
                if torch.any(gamma < 0):
                    mask_neg = gamma < 0
                    gamma[mask_neg] *= -1
                
                gamma = gamma.pow(1/order)
                if mask_neg != None:
                    gamma[mask_neg] *= -1
                # gamma = F.tanh(gamma)

                # print(f'[mu] mean:{mu.mean()}, min:{mu.min()}, max:{mu.max()}')
                # print(f'[sigma] mean:{sigma.mean()}, min:{sigma.min()}, max:{sigma.max()}')
                # print(f'[{order} order gamma] mean:{gamma.mean()}, min:{gamma.min()}, max:{gamma.max()}')
                out_1hop.append(gamma)
        # assert len(out) == 2
        # return [sigma]
        if self.mixhop:
            out = (out_1hop, out_2hop)
        else:
            out = (out_1hop, None)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class myGraphSAGE(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, moment=1, hidden=16, mode='sum', use_norm=False, mixhop=True, use_adj_norm=False, use_adj_cache=True, device=None, use_center_moment=None):
        super(myGraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.adj_t_cache = None
        self.use_adj_cache = use_adj_cache
        self.use_adj_norm = use_adj_norm
        print('moment:', moment)
        if mode in ['attention']:
            self.out_layer_mode = 'mean' # 'mean'
            self.out_layer_moment = moment
        else:
            self.out_layer_mode = mode
            self.out_layer_moment = moment
        # hidden *= moment # for scheme 1
        # self.input_layer = Linear(dataset.num_features, 128)

        if layer_num == 1:
            self.convs.append(
                my_SAGEConv(dataset.num_features, dataset.num_classes, use_norm=use_norm, moment=moment, mode=mode, mixhop=mixhop, N=dataset.num_features, use_adj_norm=self.use_adj_norm, device=device, use_center_moment=use_center_moment)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(my_SAGEConv(dataset.num_features, hidden, use_norm=use_norm, moment=moment, mode=mode, mixhop=mixhop, N=dataset.num_features, use_adj_norm=self.use_adj_norm, device=device, use_center_moment=use_center_moment))
                    # self.convs.append(my_SAGEConv(dataset.num_features, hidden, use_norm=use_norm, moment=1, mode='mean', mixhop=mixhop, N=dataset.num_features, use_adj_norm=self.use_adj_norm, device=device, use_center_moment=False))
                elif num == layer_num - 1:
                    self.convs.append(my_SAGEConv(hidden, dataset.num_classes, use_norm=use_norm, moment=self.out_layer_moment, mode=self.out_layer_mode, mixhop=mixhop, N=dataset.num_features, use_adj_norm=self.use_adj_norm,  device=device, use_center_moment=use_center_moment))
                    # self.convs.append(my_SAGEConv(hidden, dataset.num_classes, use_norm=use_norm, moment=1, mode='mean', mixhop=mixhop, N=dataset.num_features, use_adj_norm=self.use_adj_norm,  device=device, use_center_moment=False))
                    # self.convs.append(my_SAGEConv(hidden, hidden, moment=moment, mode=mode, mixhop=mixhop, N=dataset.num_features, device=device))
                else:
                    # self.convs.append(SAGEConv(hidden , hidden))
                    self.convs.append(my_SAGEConv(hidden, hidden, use_norm=use_norm, moment=moment, mode=mode, mixhop=mixhop, N=dataset.num_features, use_adj_norm=self.use_adj_norm, device=device, use_center_moment=use_center_moment))
        # self.out_layer = nn.Linear(hidden * 2, dataset.num_classes)
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # self.out_layer.reset_parameters()
        # nn.init.xavier_uniform_(self.out_layer.weight, gain=1.414)
    
    def get_attention(self, data):
        x, edge_index = data.x, data.edge_index
        attention_bucket = []
        if isinstance(edge_index, torch.Tensor) and (self.adj_t_cache == None or not self.use_adj_cache):
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            if self.use_adj_norm:
                self.adj_t_cache = adj_norm(self.adj_t_cache, norm='row')

        for ind, conv in enumerate(self.convs):
            assert isinstance(conv, my_SAGEConv)
            if ind == len(self.convs) -1:
                attention_bucket.append(conv.get_attention_layer(x, self.adj_t_cache, edge_row_col=edge_index))
                x, moment_list, moment_list_2hop = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)

            else:
                attention_bucket.append(conv.get_attention_layer(x, self.adj_t_cache, edge_row_col=edge_index))
                x, moment_list, moment_list_2hop = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)
                x = F.dropout(F.relu(x), p=0.5, training=self.training)
        return attention_bucket
    def forward(self, data, get_moment=False):
        x, edge_index = data.x, data.edge_index
        # out_list = []
        # enum = edge_index.shape[1]
        # adj_sp = torch.sparse.LongTensor(edge_index, torch.ones(enum).to(edge_index.device), torch.Size((x.shape[0], x.shape[0])))
        if isinstance(edge_index, torch.Tensor) and (self.adj_t_cache == None or not self.use_adj_cache):
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            if self.use_adj_norm:
                self.adj_t_cache = adj_norm(self.adj_t_cache, norm='row')
        if get_moment:
            moment_list_bucket = []
            moment_list_2hop_bucket = []
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x, moment_list, moment_list_2hop = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)

                if get_moment:
                    moment_list_bucket.append(moment_list)
                    moment_list_2hop_bucket.append(moment_list_2hop)
                # x = F.relu(x)
                # x = F.dropout(F.relu(x), p=0.5, training=self.training)
                # print(ind, x)
                # print(ind, conv.lin_l.weight.grad, conv.lin_r.weight.grad)
                # out_list.append(x)
            else:
                x, moment_list, moment_list_2hop = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)

                x = F.dropout(F.relu(x), p=0.5, training=self.training)
                # out_list.append(x)
                if get_moment:
                    moment_list_bucket.append(moment_list)
                    moment_list_2hop_bucket.append(moment_list_2hop)
        # x = self.out_layer(torch.cat(out_list, dim=-1))
        if get_moment:
            return F.log_softmax(x, dim=1), moment_list_bucket, moment_list_2hop_bucket
        return F.log_softmax(x, dim=1)
    

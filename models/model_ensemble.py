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
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing, gat_conv, gcn_conv, sage_conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import SplineConv, GATConv, SAGEConv, GCNConv
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from typing import Union, Tuple, Optional
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch_sparse


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

class moment_SAGEConv(MessagePassing):
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
                 use_center_moment = True,
                 **kwargs):  # yapf: disable
        if use_adj_norm:
            kwargs.setdefault('aggr', 'add')
        else:
            kwargs.setdefault('aggr', 'mean')
        super(moment_SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.moment = moment
        self.mode = mode
        self.mixhop = mixhop
        self.moment_att_dim = moment_att_dim
        self.use_adj_norm = use_adj_norm
        self.use_center_moment = use_center_moment
        print('mode:', mode)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        # scheme 1
        self.lin_l = Linear(in_channels[0], out_channels, bias=bias) # moment * d -> out_dim, input is the concat feat of all moments


        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
    
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_row_col:Adj=None, thres_deg=0,
                size: Size = None, get_moment=False) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out_list, out_list_2hop = self.propagate(edge_index, x=x, size=size, moment=self.moment)
        # fusion the output from different moment and different hop
        out = out_list[-1]
        out = self.lin_l(out)


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
        
        if moment == 2:
            # sigma = matmul(adj_t, (x[0]).pow(2), reduce=self.aggr)
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
        elif moment > 2:
            order = moment
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

            out_1hop.append(gamma)

        if self.mixhop:
            out = (out_1hop, out_2hop)
        else:
            out = (out_1hop, None)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class myGraphSAGE_moment(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, moment=1, hidden=16, mode='sum',  mixhop=True, use_adj_norm=False, use_adj_cache=True, use_center_moment=None, device=None):
        super(myGraphSAGE_moment, self).__init__()
        self.convs = nn.ModuleList()
        self.adj_t_cache = None
        self.use_adj_cache = use_adj_cache
        self.use_adj_norm = use_adj_norm
        print('moment:', moment)
        assert use_center_moment is not None
        if layer_num == 1:
            self.convs.append(
                moment_SAGEConv(dataset.num_features, dataset.num_classes)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(moment_SAGEConv(dataset.num_features, hidden, moment=moment, \
                        mode=mode, mixhop=mixhop, N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm, \
                            use_center_moment=use_center_moment, device=device))
                elif num == layer_num - 1:
                    self.convs.append(moment_SAGEConv(hidden, dataset.num_classes, moment=moment, mode=mode, \
                        mixhop=mixhop, N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm, \
                             use_center_moment=use_center_moment, device=device))
                else:
                    self.convs.append(moment_SAGEConv(hidden, hidden, moment=moment, mode=mode, mixhop=mixhop, \
                        N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm, use_center_moment=use_center_moment, \
                            device=device))
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    

    def forward(self, data, get_moment=False):
        x, edge_index = data.x, data.edge_index
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
            else:
                x, moment_list, moment_list_2hop = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)
                x = F.dropout(F.relu(x), p=0.5, training=self.training)
                if get_moment:
                    moment_list_bucket.append(moment_list)
                    moment_list_2hop_bucket.append(moment_list_2hop)
        if get_moment:
            return F.log_softmax(x, dim=1), moment_list_bucket, moment_list_2hop_bucket
        return F.log_softmax(x, dim=1)
    


class split_moment_SAGEConv(MessagePassing):
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
                 use_center_moment = True,
                 **kwargs):  # yapf: disable
        if use_adj_norm:
            kwargs.setdefault('aggr', 'add')
        else:
            kwargs.setdefault('aggr', 'mean')
        super(split_moment_SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.moment = moment
        self.mode = mode
        self.mixhop = mixhop
        self.moment_att_dim = moment_att_dim
        self.use_adj_norm = use_adj_norm
        self.use_center_moment = use_center_moment
        print('mode:', mode)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        # scheme 1
        self.lin_moment_list = nn.ModuleList()
        for _ in range(moment):
            self.lin_moment_list.append(Linear(in_channels[0], out_channels, bias=bias))
        print(moment, len(self.lin_moment_list))

        if self.root_weight:
            self.lin_r_list = nn.ModuleList()
            for _ in range(moment):
                self.lin_r_list.append(Linear(in_channels[1], out_channels, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        for fc in self.lin_moment_list:
            fc.reset_parameters()
            # nn.init.xavier_uniform_(fc.weight, gain=1.414)
        if self.root_weight:
            for fc in self.lin_r_list:
                fc.reset_parameters()
                # nn.init.xavier_uniform_(fc.weight, gain=1.414)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_row_col:Adj, thres_deg=0,
                size: Size = None, get_moment=False) -> Tensor:
        """"""
        
        if isinstance(x, Tensor):
            # x: OptPairTensor = (x, x)
            if len(x.shape) == 2:
                X = [x] * self.moment
                X_r = None
                if self.root_weight:
                    X_r = [x] * self.moment
            elif len(x.shape) == 3:
                X = [x[i] for i in range(x.shape[0])]
                X_r = None
                if self.root_weight:
                    X_r = [x[i] for i in range(x.shape[0])]
        

        # propagate_type: (x: OptPairTensor)
        out_list = self.propagate(edge_index, x=X, size=size, moment=self.moment)
        # fusion the output from different moment and different hop
        for idx, fc in enumerate(self.lin_moment_list):
            out_list[idx] = fc(out_list[idx])
  


        if self.root_weight and X_r is not None:
            for idx, fc in enumerate(self.lin_r_list):
                out_list[idx] += fc(X_r[idx])
                if self.normalize:
                    out_list[idx] = F.normalize(out_list[idx], p=2., dim=-1)

        return torch.stack(out_list, dim=0) # (m, N, d)
        # return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: list,
                              moment: int) -> Tensor:

        # adj_t = adj_t.set_value(None, layout=None)
        mu = matmul(adj_t, x[0], reduce=self.aggr)
        
        # sigma = matmul(adj_t, (x[0] - mu).pow(4), reduce=self.aggr)
        out_1hop = [mu]
        
        if moment >= 2:
            # sigma = matmul(adj_t, (x[0]).pow(2), reduce=self.aggr)
            if self.use_center_moment:
                sigma = matmul(adj_t, (x[1] - mu).pow(2), reduce=self.aggr)
            else:
                sigma = matmul(adj_t, (x[1]).pow(2), reduce=self.aggr)
            sigma[sigma == 0] = 1e-16
            sigma = sigma.sqrt()
            out_1hop.append(sigma)
            for order in range(3, moment+1):
                idx = order - 1
                # gamma = matmul(adj_t, (x[idx] - mu).div(sigma).pow(order), reduce=self.aggr)
                # gamma = matmul(adj_t, (x[idx] - mu).pow(order), reduce=self.aggr)
                gamma = matmul(adj_t, x[idx].pow(order), reduce=self.aggr)
                mask_neg = None
                if torch.any(gamma == 0):
                    gamma[gamma == 0] = 1e-16
                if torch.any(gamma < 0):
                    mask_neg = gamma < 0
                    gamma[mask_neg] *= -1
                
                gamma = gamma.pow(1/order)
                if mask_neg != None:
                    gamma[mask_neg] *= -1
                out_1hop.append(gamma)
        return out_1hop # (m, N, d)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class myGraphSAGE_moment_split(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, moment=1, hidden=16, mode='sum',  mixhop=True, use_adj_norm=False, use_adj_cache=True, use_center_moment=None, device=None):
        super(myGraphSAGE_moment_split, self).__init__()
        self.convs = nn.ModuleList()
        self.adj_t_cache = None
        self.use_adj_cache = use_adj_cache
        self.use_adj_norm = use_adj_norm
        self.out_layer = Linear(hidden * moment, dataset.num_classes, bias=False)
        # self.out_layer = SAGEConv(hidden * moment, dataset.num_classes)
        # self.out_layer = moment_SAGEConv(hidden * moment, dataset.num_classes, moment=1, \
        #                 mode=mode, mixhop=mixhop, N=dataset[0].x.shape[0], use_adj_norm=use_adj_norm, \
        #                     use_center_moment=False, device=device)
        print('moment:', moment)
        assert use_center_moment is not None
        if layer_num == 1:
            self.convs.append(
                moment_SAGEConv(dataset.num_features, dataset.num_classes)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(split_moment_SAGEConv(dataset.num_features, hidden, moment=moment, \
                        mode=mode, mixhop=mixhop, N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm, \
                            use_center_moment=use_center_moment, device=device))
                elif num == layer_num - 1:
                    self.convs.append(split_moment_SAGEConv(hidden, hidden, moment=moment, mode=mode, \
                        mixhop=mixhop, N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm, \
                             use_center_moment=use_center_moment, device=device))
                    # self.convs.append(split_moment_SAGEConv(hidden, dataset.num_classes, moment=moment, mode=mode, \
                    #     mixhop=mixhop, N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm, \
                    #          use_center_moment=use_center_moment, device=device))
                else:
                    self.convs.append(split_moment_SAGEConv(hidden, hidden, moment=moment, mode=mode, mixhop=mixhop, \
                        N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm, use_center_moment=use_center_moment, \
                            device=device))
    def reset_parameters(self):
        self.out_layer.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    

    def forward(self, data, get_moment=False):
        x, edge_index = data.x, data.edge_index
        if isinstance(edge_index, torch.Tensor) and (self.adj_t_cache == None or not self.use_adj_cache):
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            if self.use_adj_norm:
                self.adj_t_cache = adj_norm(self.adj_t_cache, norm='row')

        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)
                x = F.dropout(F.relu(x), p=0.5, training=self.training)
            else:
                x = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)
                x = F.dropout(F.relu(x), p=0.5, training=self.training)
        # print(1, x[0].min(), x[0].max(), x[0].mean())
        # print(2, x[1].min(), x[1].max(), x[1].mean())
        # print(3, x[2].min(), x[2].max(), x[2].mean())
        
        # x = x.mean(0)
        x = self.out_layer(torch.cat([x[idx, :, :] for idx in range(x.shape[0])], dim=-1))
        # x = self.out_layer(torch.cat([x[idx, :, :] for idx in range(x.shape[0])], dim=-1), self.adj_t_cache)
        # x, _, _ = self.out_layer(torch.cat([x[idx, :, :] for idx in range(x.shape[0])], dim=-1), self.adj_t_cache)
        return F.log_softmax(x, dim=1)
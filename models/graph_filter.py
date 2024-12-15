
from builtins import NotImplementedError
from functools import reduce
from tkinter import N
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
from torch_geometric.nn import SplineConv, GATConv, GATv2Conv, SAGEConv, GCNConv, GCN2Conv, GENConv, DeepGCNLayer, APPNP, JumpingKnowledge, GINConv
from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
import torch_sparse
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

from init import glorot, zeros

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


class FilterGraphConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0., add_self_loops: bool = True,
                 bias: bool = True, share_weights: bool = False, **kwargs):
        super(FilterGraphConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights

        self.lin_l = Linear(in_channels, heads * out_channels, bias=False,
                            weight_initializer='glorot')

        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=False,
                                weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        

        if bias and concat:
            # self.bias = Parameter(torch.Tensor(heads * out_channels))
            self.bias = Linear(in_channels, heads * out_channels, bias=True,
                                weight_initializer='glorot')
        elif bias and not concat:
            # self.bias = Parameter(torch.Tensor(out_channels))
            self.bias = Linear(in_channels, 1, bias=True,
                                weight_initializer='glorot')
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.feat_ln = nn.LayerNorm(out_channels, elementwise_affine=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        # zeros(self.bias)
        zeros(self.bias.weight)
        zeros(self.bias.bias)
        self.feat_ln.reset_parameters()
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, adj_t: torch_sparse.SparseTensor = None,
                size: Size = None, return_attention_weights: bool = None):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)


        mu_neb = matmul(adj_t, x[0], reduce='mean')

        # feat_ln
        # x_l = self.feat_ln(x_l)
        # x_r = self.feat_ln(x_r)
        out = self.propagate(edge_index, x=(x_l, x_r), mu_neb=mu_neb, size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias(x[0])

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, edge_index, x_j: Tensor, x_i: Tensor, mu_neb: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        
        # # dot_product
        # alpha = (mu_neb[edge_index[0]] * mu_neb[edge_index[1]]).sum().unsqueeze(-1)
        

        # # cosine similarity
        # alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1)
        alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1).expand(edge_index.shape[1], self.heads) # M * H
        # alpha_feat = torch.nn.CosineSimilarity(dim=-1)(x_i, x_j) # M * H
        # x = x_i + x_j
        # x = F.leaky_relu(x, self.negative_slope)
        # print((x.norm(p=2, dim=-1, keepdim=True) * self.att.norm(p=2, dim=-1, keepdim=True)).shape)
        # alpha_feat = (x * self.att).sum(dim=-1) / (x.norm(p=2, dim=-1) * self.att.norm(p=2, dim=-1))
        # alpha = (alpha_mu + alpha_feat) / 2.
        # alpha = alpha_feat
        alpha = alpha_mu

        # # attention
        # if mu_neb.device != self.att_mu_neb.device:
        #     self.att_mu_neb = self.att_mu_neb.to(mu_neb.device)
        # alpha_moments.append((F.leaky_relu(mu_neb[edge_index[0]] + mu_neb[edge_index[1]], self.negative_slope) * self.att_mu_neb).sum(1).unsqueeze(-1))


        # x = x_i + x_j
        # x = F.leaky_relu(x, self.negative_slope)
        # alpha = (x * self.att).sum(dim=-1)
        # alpha = softmax(alpha, index, ptr, size_i)
        # # alpha = alpha * alpha_mu
        # # alpha = alpha_mu
        # # print('alpha', alpha.min().detach().item(), alpha.max().detach().item(), alpha.mean().detach().item())



        # # mix the two coeifficients
        # gamma = 0.5
        # alpha = gamma * alpha + (1 - gamma) * alpha_m
        
        

        self._alpha = alpha
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # return x_j * alpha.unsqueeze(-1)
        return self.feat_ln(x_j) * alpha.unsqueeze(-1)


    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class FilterGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads,
                 dropout, att_dropout):
        super(FilterGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(FilterGraphConv(in_channels, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
 

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(FilterGraphConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
    
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(FilterGraphConv(hidden_channels * heads, out_channels, heads=1, dropout=att_dropout, concat=False))
    

        self.dropout = dropout
        self.adj_t_cache = None

        self.mmd_mat = None

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

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, adj_t=self.adj_t_cache)
            # x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, adj_t=self.adj_t_cache)
        return x.log_softmax(dim=-1)

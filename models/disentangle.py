
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
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_scatter import scatter
from torch_geometric.nn.inits import glorot, zeros
from torch_sparse import coalesce, spspmm
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.norm import DiffGroupNorm
from torch.nn import BatchNorm1d#, Linear
from torch_geometric.nn.dense.linear import Linear

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

    def __init__(self, in_channels: int, out_channels: int, heads: int = 2,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0., add_self_loops: bool = True,
                 bias: bool = True, share_weights: bool = False, **kwargs):
        super(FilterGraphConv, self).__init__(aggr='mean',node_dim=0, **kwargs)
        #super().__init__(aggr='mean')


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights

        self.scale = Linear(out_channels, out_channels, bias=False,
                            weight_initializer='glorot')

        self.lin_l = Linear(in_channels, heads * out_channels, bias=False,
                            weight_initializer='glorot')
        self.res = Linear(in_channels, heads * out_channels, bias=False,
                            weight_initializer='glorot')

        #self.lin_l = Linear(in_channels, heads * out_channels, bias=True,
                            #weight_initializer='glorot')

        #if share_weights:
            #self.lin_r = self.lin_l
        #else:
            #self.lin_r = Linear(in_channels, heads * out_channels, bias=True,
                               # weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1))
        nn.init.constant_(self.att, 2)
        #self.sim
        
        self.mess = Linear(out_channels*2, out_channels, bias=True,
                                weight_initializer='glorot')

        #self.mess1 = Linear(out_channels, out_channels, bias=True, weight_initializer='glorot')                     
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
        self.scale.reset_parameters()
        self.res.reset_parameters()
        self.mess.reset_parameters()
        #self.mess1.reset_parameters()
        #self.lin_r.reset_parameters()
        #glorot(self.att)
        # zeros(self.bias)
        zeros(self.bias.weight)
        zeros(self.bias.bias)
        self.feat_ln.reset_parameters()



    def forward(self, x: Union[Tensor, PairTensor], alpha, edge_index: Adj, norm, adj_t: torch_sparse.SparseTensor = None,
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
                x_r = self.lin_l(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_l(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        #if self.add_self_loops:
            


        # feat_ln
        # x_l = self.feat_ln(x_l)
        # x_r = self.feat_ln(x_r)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha = alpha, size=size, norm = norm)

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
            return out #+ self.res(x[1])

    def message(self, edge_index, x_j: Tensor, x_i: Tensor, alpha,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], norm) -> Tensor:
        
        # # dot_product
        # alpha = (mu_neb[edge_index[0]] * mu_neb[edge_index[1]]).sum().unsqueeze(-1)
        

        # # cosine similarity
        # alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1)
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]] + self.biasatt(mu_neb[edge_index[0]]), mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[0]])).unsqueeze(-1).expand(edge_index.shape[1], self.heads) # M * H
        # alpha_feat = torch.nn.CosineSimilarity(dim=-1)(x_i, x_j) # M * H
        #x = torch.cat((x_i,x_j * alpha),-1)
        #x = torch.cat((x_i,x_j),-1)
        #x = self.mess(x)
        #x = F.leaky_relu(x, self.negative_slope)
        #x = F.tanh(x)
        #x = self.feat_ln(x)
        #x = self.mess1(x)
        # print((x.norm(p=2, dim=-1, keepdim=True) * self.att.norm(p=2, dim=-1, keepdim=True)).shape)
        # alpha_feat = (x * self.att).sum(dim=-1) / (x.norm(p=2, dim=-1) * self.att.norm(p=2, dim=-1))
        # alpha = (alpha_mu + alpha_feat) / 2.
        # alpha = alpha_feat
        #alpha = alpha_mu
        row,col = edge_index
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
        # mix the two coeifficients
        # gamma = 0.5
        # alpha = gamma * alpha + (1 - gamma) * alpha_m
        #self._alpha = alpha
        
        #alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # return x_j * alpha.unsqueeze(-1)
        #return self.feat_ln(x_j) * alpha
        return x_j * F.relu(alpha.detach() - 0.5)* 2
        #return x_j * F.relu(alpha.detach() )
        #return x_j * (alpha.detach() - 0.5)* 2
        #* (1/(F.relu(alpha.detach() - 0.5) + 0.0005))#x_j * F.relu(alpha - 0.5)*2# #* (2 * alpha -1)
        #return (x_j * alpha) / (norm[col])
        #return x_j

class FilterGraphConv1(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: int, out_channels: int, heads: int = 2,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0., add_self_loops: bool = True,
                 bias: bool = True, share_weights: bool = False, **kwargs):
        super(FilterGraphConv1, self).__init__(aggr='mean', node_dim=0, **kwargs)
        #super().__init__(aggr='mean')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.scale = Linear(int(self.out_channels/self.heads), int(self.out_channels/self.heads), bias=False,
                            weight_initializer='glorot')
        self.mess = Linear(2*int(self.out_channels/self.heads), int(self.in_channels/self.heads), bias=True,
                                weight_initializer='glorot')

        
        self.res = Linear(in_channels, out_channels, bias=False,
                            weight_initializer='glorot')
        self.lin_l = Linear(in_channels, heads * out_channels, bias=False,
                            weight_initializer='glorot')

        #self.lin_l = Linear(in_channels, heads * out_channels, bias=True,
                            #weight_initializer='glorot')

        #if share_weights:
            #self.lin_r = self.lin_l
        #else:
            #self.lin_r = Linear(in_channels, heads * out_channels, bias=True,
                               # weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1))
        nn.init.constant_(self.att, 2)
        #self.sim
        
        

        if bias and concat:
            # self.bias = Parameter(torch.Tensor(heads * out_channels))
            self.bias = Linear(in_channels, in_channels, bias=True,
                                weight_initializer='glorot')
        elif bias and not concat:
            # self.bias = Parameter(torch.Tensor(out_channels))
            self.bias = Linear(in_channels, 1, bias=True,
                                weight_initializer='glorot')
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.feat_ln = nn.LayerNorm(int(self.in_channels/self.heads), elementwise_affine=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.scale.reset_parameters()
        self.res.reset_parameters()
        #self.lin_r.reset_parameters()
        #glorot(self.att)
        # zeros(self.bias)
        zeros(self.bias.weight)
        zeros(self.bias.bias)
        self.feat_ln.reset_parameters()
        #zeros(self.mess.weight)
        #zeros(self.mess.bias)
        self.mess.reset_parameters()



    def forward(self, x: Union[Tensor, PairTensor], alpha, edge_index: Adj, norm, adj_t: torch_sparse.SparseTensor = None,
                size: Size = None, return_attention_weights: bool = None):
        H, C = self.heads, int(self.out_channels/self.heads)

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = x.view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = x.view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = x_l.view(-1, H, C)
            if x_r is not None:
                x_r = x_r.view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        #if self.add_self_loops:
            


        # feat_ln
        # x_l = self.feat_ln(x_l)
        # x_r = self.feat_ln(x_r)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha = alpha, size=size, norm = norm)
        #out = self.feat_ln(out)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.in_channels)
        else:
            out = out.mean(dim=1)

        #if self.bias is not None:
            #out += self.bias(x[0])

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out#+self.res(x[1])

    def message(self, edge_index, x_j: Tensor, x_i: Tensor, alpha,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], norm) -> Tensor:
        # # cosine similarity
        # alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1)
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]] + self.biasatt(mu_neb[edge_index[0]]), mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[0]])).unsqueeze(-1).expand(edge_index.shape[1], self.heads) # M * H
        # alpha_feat = torch.nn.CosineSimilarity(dim=-1)(x_i, x_j) # M * H
        #x = x_i + x_j
        #x = F.leaky_relu(x, self.negative_slope)
        #x = torch.cat((x_i,x_j * alpha),-1)
        #x = torch.cat((x_i,x_j),-1)
        #x = self.mess(x)
        #x = F.leaky_relu(x, self.negative_slope)
        #x = self.feat_ln(x)
        # print((x.norm(p=2, dim=-1, keepdim=True) * self.att.norm(p=2, dim=-1, keepdim=True)).shape)
        # alpha_feat = (x * self.att).sum(dim=-1) / (x.norm(p=2, dim=-1) * self.att.norm(p=2, dim=-1))
        # alpha = (alpha_mu + alpha_feat) / 2.
        # alpha = alpha_feat
        #alpha = alpha_mu
        row, col = edge_index
        #deg = degree(row, size[0], dtype=x_j.dtype)
        #deg_inv_sqrt = deg.pow(-0.5)
        #norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        
        #x_j =  x_j.view(-1, out_dim * heads)
        #x_j = norm.view(-1,1) * x_j
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
        # mix the two coeifficients
        # gamma = 0.5
        # alpha = gamma * alpha + (1 - gamma) * alpha_m
        #self._alpha = alpha
        
        #alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # return x_j * alpha.unsqueeze(-1)
       # return self.feat_ln(x_j) * alpha
        #return x
        # return x_j * alpha.unsqueeze(-1)
        #return x_j
        return x_j * F.relu(alpha.detach() - 0.5)*2#* (1/(F.relu(alpha.detach() - 0.5) + 0.0005)) #* F.relu(alpha.detach() - 0.5)*2#x_j * F.relu(alpha - 0.5)*2# #* (2 * alpha -1)
        #return x_j * F.relu(alpha.detach() )
        #return (x_j * alpha) / (norm[col])


def __repr__(self):
    return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class FilterGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads,
                 dropout, att_dropout):
        super(FilterGNN, self).__init__()
        


        self.heads = heads
        self.lin = nn.Sequential(
            nn.Linear(in_channels, 512, bias=True),
                            #weight_initializer='glorot'),
            nn.Tanh(),  
            #nn.BatchNorm1d(512),     
            #nn.Linear(256, 256, bias=True),
                            #weight_initializer='glorot'),

            #nn.Tanh(),
            nn.Linear(512, 256, bias=True),
            nn.Tanh(), 
            nn.Linear(256, 128, bias=True),


            
        )

        
        for m in self.lin:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

       
        self.convs = torch.nn.ModuleList()
        self.convs.append(FilterGraphConv1(128, 128, heads=heads, dropout=att_dropout, concat=True))
        self.agg = nn.Linear(2, 1, bias=False)
        
        nn.init.kaiming_normal_(self.agg.weight)
        #nn.init.constant_(self.agg.bias, 0)                    
        #self.convs.append(FilterGraphConv(in_channels, 128, heads=heads, dropout=att_dropout, concat=True))
        #self.first = firstfilter(dropout=att_dropout, concat=True)

        self.biasatt = nn.Sequential(
            nn.Linear(128*2, 64*2, bias=True),
                            #weight_initializer='glorot'),
            nn.Tanh(),  
            #nn.BatchNorm1d(128), 
             
            nn.Linear(64*2, 128*2, bias=True),
                            #weight_initializer='glorot'),
            
        )

        for m in self.biasatt:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.biasatt1 = nn.Sequential(
            nn.Linear(128, 64, bias=True),
                            #weight_initializer='glorot'),
            nn.Tanh(),  
            #nn.BatchNorm1d(128), 
             
            nn.Linear(64, 128, bias=True),
                            #weight_initializer='glorot'),
            
        )

        for m in self.biasatt1:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


        self.lin1 = nn.Sequential(
            nn.Linear(128, 64, bias=True),
                            #weight_initializer='glorot'),
            nn.Tanh(),  
            nn.BatchNorm1d(64),     
            #nn.Linear(256, 256, bias=True),
                            #weight_initializer='glorot'),

            #nn.Tanh(),
            #nn.Linear(512, 256, bias=True),
            #nn.Tanh(), 
            nn.Linear(64, 128, bias=True),


            
        )
        
        #self.beta = Parameter(torch.empty(in_channels))
        #torch.nn.init.normal_(self.beta, mean=0, std=1)

        self.theta = Parameter(torch.empty(5))
        torch.nn.init.normal_(self.theta, mean=1, std=0.5)
        
        for m in self.lin1:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        

        #self.mlp = nn.Sequential(
           # nn.Linear(128*4, 128, bias=True),
                            #weight_initializer='glorot'),
           # nn.Tanh(),  
            #nn.BatchNorm1d(512),     
            #nn.Linear(256, 256, bias=True),
                            #weight_initializer='glorot'),

            #nn.Tanh(),
           # nn.Linear(128, 1, bias=True),


            
        #)
        #for m in self.mlp:
           # if isinstance(m, nn.Linear):
            #    nn.init.kaiming_normal_(m.weight)
            #    nn.init.constant_(m.bias, 0)

        self.trans = nn.Linear(128, 1, bias=True)
        
        nn.init.kaiming_normal_(self.trans.weight)
        nn.init.constant_(self.trans.bias, 0)    

        #self.bns = torch.nn.ModuleList()
        #self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        

        for _ in range(num_layers-2):
            self.convs.append(FilterGraphConv(128, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
    
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        #self.convs.append(FilterGraphConv(256, 128, heads, dropout=att_dropout, concat=False))
        #self.convs.append(FilterGraphConv(heads * 128, out_channels, heads=1, dropout=att_dropout, concat=False))
        self.convs.append(FilterGraphConv(128, out_channels, heads=1, dropout=att_dropout, concat=False))

        self.dropout = dropout
        self.adj_t_cache = None

        self.mmd_mat = None

        self.lin2 = nn.Sequential(
            nn.Linear(in_channels, 512, bias=True),
                            #weight_initializer='glorot'),
            #nn.ReLU(), 
            
            nn.Tanh(),
            DiffGroupNorm(512,4) ,
            #nn.BatchNorm1d(512),     
            #nn.Linear(256, 256, bias=True),
                            #weight_initializer='glorot'),

            #nn.Tanh(),
            #nn.Linear(512, 256, bias=True),
            #nn.Tanh(), 
            nn.Linear(512, 256, bias=True),
            

            #nn.ReLU(),
            
            nn.Tanh(),
            DiffGroupNorm(256,4) ,
            #nn.BatchNorm1d(256),

            nn.Linear(256, 128, bias=True),
            #nn.ReLU(),
            
            #nn.Tanh(),
            #DiffGroupNorm(128,4) ,


            #nn.BatchNorm1d(128),

            #nn.Linear(128, 128, bias=True),
            #DiffGroupNorm(128,out_channels) ,


            
        )

        for m in self.lin2:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)



        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        #for bn in self.bns:
           # bn.reset_parameters()
        #self.agg.reset_parameters()

    def similarity(self, edge_index, mu, mu_neb,head) -> Tensor:
        
        # # dot_product
        #alpha = ((mu_neb[edge_index[0]] + self.biasatt(mu_neb[edge_index[0]]) * (mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[0]])))).sum().unsqueeze(-1)
        #alpha = ((mu_neb[edge_index[0]]+ self.biasatt(mu_neb[edge_index[0]]))  * (mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[0]]))).sum().unsqueeze(-1).expand(edge_index.shape[1], head) 
        #alpha = ((mu_neb[edge_index[0]])  * (mu_neb[edge_index[1]])).sum().unsqueeze(-1).expand(edge_index.shape[1], head) 
        # # cosine similarity
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1)#.expand(edge_index.shape[1], head)
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]] + self.biasatt(torch.cat((mu_neb[edge_index[1]],mu_neb[edge_index[0]]),-1)), mu_neb[edge_index[1]]+ self.biasatt(torch.cat((mu_neb[edge_index[1]],mu_neb[edge_index[0]]),-1))).unsqueeze(-1)
         # M * H
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]] + self.out(mu_neb[edge_index[1]]), mu_neb[edge_index[1]] + self.out(mu_neb[edge_index[1]])).unsqueeze(-1).expand(edge_index.shape[1], head)
        
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(self.biasatt(mu_neb[edge_index[1]]), self.biasatt(mu_neb[edge_index[1]])).unsqueeze(-1)
        #mu_neb = torch.cat((mu,mu_neb),-1)
        #mu_neb = self.lin2(mu_neb)
        #mu = self.lin1(mu)
        mu = self.lin3(mu)
        mu_neb = self.lin4(mu_neb)
        mu_neb = torch.cat((mu,mu_neb),-1)
        alpha = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]]+ self.biasatt(mu_neb[edge_index[1]]), mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[1]])).unsqueeze(-1)
        #nu = torch.nn.CosineSimilarity(dim=1)(mu[edge_index[0]]+ self.biasatt1(mu[edge_index[1]]), mu[edge_index[1]]+ self.biasatt1(mu[edge_index[1]])).unsqueeze(-1)
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1)
        #nu = torch.nn.CosineSimilarity(dim=1)(mu[edge_index[0]], mu[edge_index[1]]).unsqueeze(-1)
        #alpha = torch.cat((alpha_mu,nu),-1)
        # alpha_feat = torch.nn.CosineSimilarity(dim=-1)(x_i, x_j) # M * H
        # print((x.norm(p=2, dim=-1, keepdim=True) * self.att.norm(p=2, dim=-1, keepdim=True)).shape)
        # alpha_feat = (x * self.att).sum(dim=-1) / (x.norm(p=2, dim=-1) * self.att.norm(p=2, dim=-1))
        # alpha = (alpha_mu + alpha_feat) / 2.
        # alpha = alpha_feat
        #alpha = self.agg(alpha)
        #self._alpha = alpha_mu
        #alpha = F.relu(alpha - 0.2) + 0.2
        #alpha = self.linear(mu_neb[edge_index[1]]) * F.relu(F.dropout(alpha, p=self.dropout, training=self.training)) 
        #alpha = (F.dropout(alpha, p=self.dropout, training=self.training))/((F.dropout(alpha, p=self.dropout, training=self.training)).abs() + 0.00005)
        #alpha1 = F.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training)*self.theta)
        alpha1 = (F.dropout(alpha, p=self.dropout, training=self.training) + 1)/2
        beta = torch.cat(
                    (1-alpha1, alpha1), 1)
        # return x_j * alpha.unsqueeze(-1)
        return alpha.expand(edge_index.shape[1], head).unsqueeze(-1),beta


    def similarity1(self, edge_index, mu, mu_neb,head) -> Tensor:
        
        # # dot_product
        #alpha = ((mu_neb[edge_index[0]] + self.biasatt(mu_neb[edge_index[0]]) * (mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[0]])))).sum().unsqueeze(-1)
        #alpha = ((mu_neb[edge_index[0]]+ self.biasatt(mu_neb[edge_index[0]]))  * (mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[0]]))).sum().unsqueeze(-1).expand(edge_index.shape[1], head) 
        #alpha = ((mu_neb[edge_index[0]])  * (mu_neb[edge_index[1]])).sum().unsqueeze(-1).expand(edge_index.shape[1], head) 
        # # cosine similarity
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1)#.expand(edge_index.shape[1], head)
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]] + self.biasatt(torch.cat((mu_neb[edge_index[1]],mu_neb[edge_index[0]]),-1)), mu_neb[edge_index[1]]+ self.biasatt(torch.cat((mu_neb[edge_index[1]],mu_neb[edge_index[0]]),-1))).unsqueeze(-1)
         # M * H
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]] + self.out(mu_neb[edge_index[1]]), mu_neb[edge_index[1]] + self.out(mu_neb[edge_index[1]])).unsqueeze(-1).expand(edge_index.shape[1], head)
        
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(self.biasatt(mu_neb[edge_index[1]]), self.biasatt(mu_neb[edge_index[1]])).unsqueeze(-1)
        #mu_neb = torch.cat((mu,mu_neb),-1)
        mu_neb = mu
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]]+ self.biasatt(mu_neb[edge_index[1]]), mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[1]])).unsqueeze(-1)
        alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]]+ self.biasatt(mu_neb[edge_index[1]]), mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[1]])).unsqueeze(-1)
        # alpha_feat = torch.nn.CosineSimilarity(dim=-1)(x_i, x_j) # M * H
        # print((x.norm(p=2, dim=-1, keepdim=True) * self.att.norm(p=2, dim=-1, keepdim=True)).shape)
        # alpha_feat = (x * self.att).sum(dim=-1) / (x.norm(p=2, dim=-1) * self.att.norm(p=2, dim=-1))
        # alpha = (alpha_mu + alpha_feat) / 2.
        # alpha = alpha_feat
        alpha = alpha_mu
        #self._alpha = alpha_mu
        #alpha = F.relu(alpha - 0.2) + 0.2
        #alpha = self.linear(mu_neb[edge_index[1]]) * F.relu(F.dropout(alpha, p=self.dropout, training=self.training)) 
        #alpha = (F.dropout(alpha, p=self.dropout, training=self.training))/((F.dropout(alpha, p=self.dropout, training=self.training)).abs() + 0.00005)
        alpha = F.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training))
        beta = alpha #= torch.cat((1-alpha, alpha), 1)
        # return x_j * alpha.unsqueeze(-1)
        return alpha.expand(edge_index.shape[1], head).unsqueeze(-1),beta

    def similarity2(self, edge_index,mu_neb,head) -> Tensor:
        
        # # dot_product
        #alpha = ((mu_neb[edge_index[0]] + self.biasatt(mu_neb[edge_index[0]]) * (mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[0]])))).sum().unsqueeze(-1)
        #alpha = ((mu_neb[edge_index[0]]+ self.biasatt(mu_neb[edge_index[0]]))  * (mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[0]]))).sum().unsqueeze(-1).expand(edge_index.shape[1], head) 
        #alpha = ((mu_neb[edge_index[0]])  * (mu_neb[edge_index[1]])).sum().unsqueeze(-1).expand(edge_index.shape[1], head) 
        # # cosine similarity
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1)#.expand(edge_index.shape[1], head)
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]] + self.biasatt(torch.cat((mu_neb[edge_index[1]],mu_neb[edge_index[0]]),-1)), mu_neb[edge_index[1]]+ self.biasatt(torch.cat((mu_neb[edge_index[1]],mu_neb[edge_index[0]]),-1))).unsqueeze(-1)
         # M * H
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]] + self.out(mu_neb[edge_index[1]]), mu_neb[edge_index[1]] + self.out(mu_neb[edge_index[1]])).unsqueeze(-1).expand(edge_index.shape[1], head)
        
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(self.biasatt(mu_neb[edge_index[1]]), self.biasatt(mu_neb[edge_index[1]])).unsqueeze(-1)
        #mu_neb = torch.cat((mu,mu_neb),-1)
        mu_neb = self.lin2(mu_neb)
        mu_neb = self.lin3(mu_neb)
        alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]]+ self.biasatt1(mu_neb[edge_index[1]]), mu_neb[edge_index[1]]+ self.biasatt1(mu_neb[edge_index[1]])).unsqueeze(-1)
        # alpha_feat = torch.nn.CosineSimilarity(dim=-1)(x_i, x_j) # M * H
        # print((x.norm(p=2, dim=-1, keepdim=True) * self.att.norm(p=2, dim=-1, keepdim=True)).shape)
        # alpha_feat = (x * self.att).sum(dim=-1) / (x.norm(p=2, dim=-1) * self.att.norm(p=2, dim=-1))
        # alpha = (alpha_mu + alpha_feat) / 2.
        # alpha = alpha_feat
        alpha = alpha_mu
        #self._alpha = alpha_mu
        #alpha = F.relu(alpha - 0.2) + 0.2
        #alpha = self.linear(mu_neb[edge_index[1]]) * F.relu(F.dropout(alpha, p=self.dropout, training=self.training)) 
        #alpha = (F.dropout(alpha, p=self.dropout, training=self.training))/((F.dropout(alpha, p=self.dropout, training=self.training)).abs() + 0.00005)
        alpha = F.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training))
        beta = torch.cat(
                    (1-alpha, alpha), 1)
        # return x_j * alpha.unsqueeze(-1)
        return alpha.expand(edge_index.shape[1], head).unsqueeze(-1),beta


    def mlpsim(self, edge_index, mu, mu_neb,head) -> Tensor:
    
        # # dot_product
        #alpha = ((mu_neb[edge_index[0]] + self.biasatt(mu_neb[edge_index[0]]) * (mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[0]])))).sum().unsqueeze(-1)
        #alpha = ((mu_neb[edge_index[0]]+ self.biasatt(mu_neb[edge_index[0]]))  * (mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[0]]))).sum().unsqueeze(-1).expand(edge_index.shape[1], head) 
        #alpha = ((mu_neb[edge_index[0]])  * (mu_neb[edge_index[1]])).sum().unsqueeze(-1).expand(edge_index.shape[1], head) 
        # # cosine similarity
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1)#.expand(edge_index.shape[1], head)
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]] + self.biasatt(torch.cat((mu_neb[edge_index[1]],mu_neb[edge_index[0]]),-1)), mu_neb[edge_index[1]]+ self.biasatt(torch.cat((mu_neb[edge_index[1]],mu_neb[edge_index[0]]),-1))).unsqueeze(-1)
            # M * H
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]] + self.out(mu_neb[edge_index[1]]), mu_neb[edge_index[1]] + self.out(mu_neb[edge_index[1]])).unsqueeze(-1).expand(edge_index.shape[1], head)
        
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(self.biasatt(mu_neb[edge_index[1]]), self.biasatt(mu_neb[edge_index[1]])).unsqueeze(-1)
        #mu_neb = torch.cat((mu,mu_neb),-1)

        #alpha_mu = self.trans(mu_neb[edge_index[0]]) + self.trans(mu_neb[edge_index[1]])
        #mu_neb = torch.cat((mu,mu_neb),-1)
        #alpha_mu = self.mlp(torch.cat((mu_neb[edge_index[0]],mu_neb[edge_index[1]]),-1))
        # alpha_feat = torch.nn.CosineSimilarity(dim=-1)(x_i, x_j) # M * H
        # print((x.norm(p=2, dim=-1, keepdim=True) * self.att.norm(p=2, dim=-1, keepdim=True)).shape)
        # alpha_feat = (x * self.att).sum(dim=-1) / (x.norm(p=2, dim=-1) * self.att.norm(p=2, dim=-1))
        # alpha = (alpha_mu + alpha_feat) / 2.
        # alpha = alpha_feat
        alpha = alpha_mu
        #self._alpha = alpha_mu
        #alpha = F.relu(alpha - 0.2) + 0.2
        #alpha = self.linear(mu_neb[edge_index[1]]) * F.relu(F.dropout(alpha, p=self.dropout, training=self.training)) 
        #alpha = (F.dropout(alpha, p=self.dropout, training=self.training))/((F.dropout(alpha, p=self.dropout, training=self.training)).abs() + 0.00005)
        alpha = F.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training))
        beta = torch.cat(
                    (1-alpha, alpha), 1)
        # return x_j * alpha.unsqueeze(-1)
        return alpha.expand(edge_index.shape[1], head).unsqueeze(-1),beta

    def forward(self, data,alpha):
        x, edge_index,label,val_mask, test_mask = data.x, data.edge_index,data.y, data.val_mask, data.test_mask
        
        assert isinstance(edge_index, torch.Tensor)
        if  self.adj_t_cache == None:
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))

        adj_t=self.adj_t_cache
        #y = self.lin(x)
        #y = y.detach()
        #y_hat = label
        #z = self.lin2(x)

        #z=x
        #x= self.lin(x)
        #y1 = self.lin1(x)
        #x=x.detach()
        #z = self.lin2(x)

        #mu_neb = matmul(adj_t, z, reduce='mean')
        edge_indexp = data.edge_index
        #N = data.num_nodes

        #value = edge_indexp.new_ones((edge_indexp.size(1), ), dtype=torch.float)

        #index, value = spspmm(edge_indexp, value, edge_indexp, value, N, N, N)
        #value.fill_(0)
        #index, value = remove_self_loops(index, value)

        #edge_indexp = torch.cat([edge_indexp, index], dim=1)
        
        #edge_indexp, _ = coalesce(edge_indexp, None, N, N)
       
        
        if isinstance(edge_indexp, Tensor):
                num_nodes = x.size(0)
                edge_indexp1, _ = remove_self_loops(edge_indexp)
                edge_indexp1, _ = add_self_loops(edge_indexp, num_nodes=num_nodes)
        elif isinstance(edge_indexp, SparseTensor):
            edge_indexp1 = set_diag(edge_indexp)
        
        #_,beta = self.similarity(data.edge_index,x, mu_neb, self.heads)

        #_,beta = self.similarity(edge_indexp,z, mu_neb, self.heads)
        #alpha,_ = self.similarity(edge_indexp1,z, mu_neb, self.heads)

        #_,beta2 = self.similarity2(edge_indexp,mu_neb, self.heads)
        #alpha,_ = self.similarity2(edge_index, mu_neb, self.heads)
        row, col = edge_index
        norm = 1 #scatter(alpha, row, dim=0, reduce="sum") 
        #x= self.lin1(x)
        x=self.lin2(x)
        #x = y
        for i, conv in enumerate(self.convs[0:-1]):
            x = conv(x, alpha, edge_indexp1, norm, adj_t=self.adj_t_cache)
            #x = self.bns[i](x)
            x = F.tanh(x)
            #x= nn.BatchNorm1d(64)(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, alpha, edge_indexp1, norm, adj_t=self.adj_t_cache)
        return x.log_softmax(dim=-1)#,beta


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads,
                 dropout, att_dropout):
        super(MLP, self).__init__()


        self.heads = heads
        self.lin = nn.Sequential(
            nn.Linear(in_channels, 512, bias=True),
                            #weight_initializer='glorot'),
            nn.Tanh(),  
            #nn.BatchNorm1d(512),     
            #nn.Linear(256, 256, bias=True),
                            #weight_initializer='glorot'),

            #nn.Tanh(),
            nn.Linear(512, 256, bias=True),
            nn.Tanh(), 
            nn.Linear(256, 128, bias=True),


            
        )

        
        for m in self.lin:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.lin2 = Linear(128, out_channels, bias=True, weight_initializer='glorot')
        self.dropout = dropout
        self.reset_parameters()
        

        

    def reset_parameters(self):
        self.lin2.reset_parameters()
        

    

    def forward(self, data):
        x = data.x
        x = self.lin(x)
            #x = self.bns[i](x)
        x = F.tanh(x)
            #x= nn.BatchNorm1d(64)(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x.log_softmax(dim=-1)
    # def get_emb(self, data)


class sim(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads,
                 dropout, att_dropout):
        super(sim, self).__init__()




        self.heads = heads
        
        #nn.init.constant_(self.agg.bias, 0)                    
        #self.convs.append(FilterGraphConv(in_channels, 128, heads=heads, dropout=att_dropout, concat=True))
        #self.first = firstfilter(dropout=att_dropout, concat=True)

        self.biasatt = nn.Sequential(
            nn.Linear(128*2, 64*2, bias=True),
                            #weight_initializer='glorot'),
            nn.Tanh(),  
            #nn.BatchNorm1d(128), 
             
            nn.Linear(64*2, 128*2, bias=True),
                            #weight_initializer='glorot'),
            
        )

        for m in self.biasatt:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


        self.biasatt3 = nn.Sequential(
            nn.Linear(128, 64, bias=True),
                            #weight_initializer='glorot'),
            nn.Tanh(),  
            #nn.BatchNorm1d(128), 
             
            nn.Linear(64, 128, bias=True),
                            #weight_initializer='glorot'),
            
        )

        for m in self.biasatt3:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.biasatt1 = nn.Sequential(
            nn.Linear(128*4, 64*4, bias=True),
                            #weight_initializer='glorot'),
            nn.Tanh(),  
            #nn.BatchNorm1d(128), 
             
            nn.Linear(64*4, 128*2, bias=True),
                            #weight_initializer='glorot'),
            
        )

        for m in self.biasatt1:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)



        self.lin2 = nn.Sequential(
            nn.Linear(in_channels, 512, bias=True),
                            #weight_initializer='glorot'),
            #nn.ReLU(), 
            
            nn.Tanh(),
            DiffGroupNorm(512,4) ,
            #nn.BatchNorm1d(512),     
            #nn.Linear(256, 256, bias=True),
                            #weight_initializer='glorot'),

            #nn.Tanh(),
            #nn.Linear(512, 256, bias=True),
            #nn.Tanh(), 
            nn.Linear(512, 256, bias=True),
            

            #nn.ReLU(),
            
            nn.Tanh(),
            DiffGroupNorm(256,4) ,
            #nn.BatchNorm1d(256),

            nn.Linear(256, 128, bias=True),
            #nn.ReLU(),
            
            nn.Tanh(),
            DiffGroupNorm(128,4) ,


            #nn.BatchNorm1d(128),

            nn.Linear(128, 128, bias=True),
            #DiffGroupNorm(128,out_channels) ,


            
        )
        
        for m in self.lin2:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


        self.lin5 = nn.Sequential(
            nn.Linear(in_channels, 512, bias=False),
                            #weight_initializer='glorot'),
            nn.Tanh(),  
            #nn.BatchNorm1d(512),     
            #nn.Linear(256, 256, bias=True),
                            #weight_initializer='glorot'),

            #nn.Tanh(),
            #nn.Linear(512, 256, bias=True),
            #nn.Tanh(), 
            nn.Linear(512, 256, bias=False),

            nn.Tanh(),
            #nn.BatchNorm1d(256),

            nn.Linear(256, 128, bias=False),


            nn.BatchNorm1d(128),

            nn.Linear(128, 128, bias=False),



            
        )
        
        for m in self.lin5:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                #nn.init.constant_(m.bias, 0)

        self.lin3 = nn.Sequential(
            nn.Linear(128, 64, bias=False),
                            #weight_initializer='glorot'),
            #DiffGroupNorm(64,out_channels) ,
            nn.Tanh(),  
            #DiffGroupNorm(64,4) ,
            nn.LayerNorm(64),
            #nn.BatchNorm1d(64),     
            #nn.Linear(256, 256, bias=True),
                            #weight_initializer='glorot'),

            #nn.Tanh(),
            #nn.Linear(512, 256, bias=True),
            #nn.Tanh(), 
            nn.Linear(64, 128, bias=False),
            #DiffGroupNorm(128,out_channels) ,


            
        )
        
        for m in self.lin3:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                #nn.init.constant_(m.bias, 0)

        self.lin4 = nn.Sequential(
            nn.Linear(128, 64, bias=False),
                            #weight_initializer='glorot'),
            nn.Tanh(),  
            #DiffGroupNorm(64,4) ,
            #nn.LayerNorm(64),
            #nn.BatchNorm1d(64),     
            #nn.Linear(256, 256, bias=True),
                            #weight_initializer='glorot'),

            #nn.Tanh(),
            #nn.Linear(512, 256, bias=True),
            #nn.Tanh(), 
            nn.Linear(64, 128, bias=False),


            
        )
        
        for m in self.lin4:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        self.convs = torch.nn.ModuleList()
        self.convs.append(FilterGraphConv1(128, 128, heads=heads, dropout=att_dropout, concat=True))

        
        #self.mlp = nn.Sequential(
           # nn.Linear(128*4, 128, bias=True),
                            #weight_initializer='glorot'),
           # nn.Tanh(),  
            #nn.BatchNorm1d(512),     
            #nn.Linear(256, 256, bias=True),
                            #weight_initializer='glorot'),

            #nn.Tanh(),
           # nn.Linear(128, 1, bias=True),


            
        #)
        #for m in self.mlp:
           # if isinstance(m, nn.Linear):
            #    nn.init.kaiming_normal_(m.weight)
            #    nn.init.constant_(m.bias, 0)

        #self.trans = nn.Linear(128, 1, bias=True)
        
        #nn.init.kaiming_normal_(self.trans.weight)
        #nn.init.constant_(self.trans.bias, 0)  
        # 
        self.theta = Parameter(torch.empty(1))
        torch.nn.init.normal_(self.theta, mean=5, std=0.001)  

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers-2):
            self.convs.append(FilterGraphConv(128, hidden_channels, heads=heads, dropout=att_dropout, concat=True))
    
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        #self.convs.append(FilterGraphConv(256, 128, heads, dropout=att_dropout, concat=False))
        #self.convs.append(FilterGraphConv(heads * 128, out_channels, heads=1, dropout=att_dropout, concat=False))
        self.convs.append(FilterGraphConv(128, out_channels, heads=1, dropout=att_dropout, concat=False))

        self.dropout = dropout
        self.adj_t_cache = None

        self.mmd_mat = None
        #self.reset_parameters()
     

    #def reset_parameters(self):
    #    self.lin2.reset_parameters()

    def similarity(self, edge_index, mu, mu_neb,head) -> Tensor:
        
        #mu_neb = torch.cat((mu,mu_neb),-1)
        #mu_neb = self.lin2(mu_neb)
        #mu = self.lin1(mu)

        mu = self.lin3(mu)
        mu_neb = self.lin4(mu_neb)
        mu_neb = torch.cat((mu,mu_neb),-1)
        alpha = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]]+ self.biasatt(mu_neb[edge_index[1]]), mu_neb[edge_index[1]]+ self.biasatt(mu_neb[edge_index[1]])).unsqueeze(-1)
        #alpha = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]]+ self.biasatt1(torch.cat((mu_neb[edge_index[0]],mu_neb[edge_index[1]]),-1)), mu_neb[edge_index[1]]+ self.biasatt1(torch.cat((mu_neb[edge_index[0]],mu_neb[edge_index[1]]),-1))).unsqueeze(-1)
        #nu = torch.nn.CosineSimilarity(dim=1)(mu[edge_index[0]]+ self.biasatt1(mu[edge_index[1]]), mu[edge_index[1]]+ self.biasatt1(mu[edge_index[1]])).unsqueeze(-1)
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1)
        #nu = torch.nn.CosineSimilarity(dim=1)(mu[edge_index[0]], mu[edge_index[1]]).unsqueeze(-1)
        #alpha = torch.cat((alpha_mu,nu),-1)
        # alpha_feat = torch.nn.CosineSimilarity(dim=-1)(x_i, x_j) # M * H
        # print((x.norm(p=2, dim=-1, keepdim=True) * self.att.norm(p=2, dim=-1, keepdim=True)).shape)
        # alpha_feat = (x * self.att).sum(dim=-1) / (x.norm(p=2, dim=-1) * self.att.norm(p=2, dim=-1))
        # alpha = (alpha_mu + alpha_feat) / 2.
        # alpha = alpha_feat
        #alpha = self.agg(alpha)
        #self._alpha = alpha_mu
        #alpha = F.relu(alpha - 0.2) + 0.2
        #alpha = self.linear(mu_neb[edge_index[1]]) * F.relu(F.dropout(alpha, p=self.dropout, training=self.training)) 
        #alpha = (F.dropout(alpha, p=self.dropout, training=self.training))/((F.dropout(alpha, p=self.dropout, training=self.training)).abs() + 0.00005)
        alpha1 = F.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training)*5)
        #alpha1 = (F.dropout(alpha, p=self.dropout, training=self.training)+1)/2
        beta = torch.cat( (1-alpha1, alpha1), 1)
        # return x_j * alpha.unsqueeze(-1)
        return alpha1.expand(edge_index.shape[1], head).unsqueeze(-1),beta

    def similarity1(self, edge_index, mu, mu_neb,head) -> Tensor:
        
        #mu_neb = torch.cat((mu,mu_neb),-1)
        #mu_neb = self.lin2(mu_neb)
        #mu = self.lin5(mu)
        #mu = self.lin3(mu)
        #mu_neb = mu
        mu_neb = self.lin4(mu_neb)
        #mu_neb = torch.cat((mu,mu_neb),-1)
        alpha = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]]+ self.biasatt3(mu_neb[edge_index[1]]), mu_neb[edge_index[1]]+ self.biasatt3(mu_neb[edge_index[1]])).unsqueeze(-1)
        #alpha = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]]+ self.biasatt1(torch.cat((mu_neb[edge_index[0]],mu_neb[edge_index[1]]),-1)), mu_neb[edge_index[1]]+ self.biasatt1(torch.cat((mu_neb[edge_index[0]],mu_neb[edge_index[1]]),-1))).unsqueeze(-1)
        #nu = torch.nn.CosineSimilarity(dim=1)(mu[edge_index[0]]+ self.biasatt1(mu[edge_index[1]]), mu[edge_index[1]]+ self.biasatt1(mu[edge_index[1]])).unsqueeze(-1)
        #alpha_mu = torch.nn.CosineSimilarity(dim=1)(mu_neb[edge_index[0]], mu_neb[edge_index[1]]).unsqueeze(-1)
        #nu = torch.nn.CosineSimilarity(dim=1)(mu[edge_index[0]], mu[edge_index[1]]).unsqueeze(-1)
        #alpha = torch.cat((alpha_mu,nu),-1)
        # alpha_feat = torch.nn.CosineSimilarity(dim=-1)(x_i, x_j) # M * H
        # print((x.norm(p=2, dim=-1, keepdim=True) * self.att.norm(p=2, dim=-1, keepdim=True)).shape)
        # alpha_feat = (x * self.att).sum(dim=-1) / (x.norm(p=2, dim=-1) * self.att.norm(p=2, dim=-1))
        # alpha = (alpha_mu + alpha_feat) / 2.
        # alpha = alpha_feat
        #alpha = self.agg(alpha)
        #self._alpha = alpha_mu
        #alpha = F.relu(alpha - 0.2) + 0.2
        #alpha = self.linear(mu_neb[edge_index[1]]) * F.relu(F.dropout(alpha, p=self.dropout, training=self.training)) 
        #alpha = (F.dropout(alpha, p=self.dropout, training=self.training))/((F.dropout(alpha, p=self.dropout, training=self.training)).abs() + 0.00005)
        alpha = F.sigmoid(F.dropout(alpha, p=self.dropout, training=self.training)*5)
        #alpha = (F.dropout(alpha, p=self.dropout, training=self.training)+1)/2
        beta = torch.cat( (1-alpha, alpha), 1)
        # return x_j * alpha.unsqueeze(-1)
        return alpha.expand(edge_index.shape[1], head).unsqueeze(-1),beta
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        assert isinstance(edge_index, torch.Tensor)
        if  self.adj_t_cache == None:
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))

        adj_t=self.adj_t_cache
        #y = self.lin(x)
        #y = y.detach()
        #y_hat = label
        z = self.lin2(x)
        #z=x
        #x= self.lin(x)
        #y1 = self.lin1(x)
        #x=x.detach()
        #z = self.lin2(x)
        mu_neb = matmul(adj_t, z, reduce='mean')
        edge_indexp = data.edge_index
        #N = data.num_nodes

        #value = edge_indexp.new_ones((edge_indexp.size(1), ), dtype=torch.float)

        #index, value = spspmm(edge_indexp, value, edge_indexp, value, N, N, N)
        #value.fill_(0)
        #index, value = remove_self_loops(index, value)

        #edge_indexp = torch.cat([edge_indexp, index], dim=1)
        
        #edge_indexp, _ = coalesce(edge_indexp, None, N, N)
       
        
        if isinstance(edge_indexp, Tensor):
                num_nodes = x.size(0)
                edge_indexp1, _ = remove_self_loops(edge_indexp)
                edge_indexp1, _ = add_self_loops(edge_indexp, num_nodes=num_nodes)
        elif isinstance(edge_indexp, SparseTensor):
            edge_indexp1 = set_diag(edge_indexp)
        
        #_,beta = self.similarity(data.edge_index,x, mu_neb, self.heads)

        #_,beta = self.similarity(edge_indexp,x, mu_neb, self.heads)
        #alpha,_ = self.similarity(edge_indexp1,x, mu_neb, self.heads)
        _,beta = self.similarity(edge_indexp,z, mu_neb, self.heads)
        alpha,_ = self.similarity(edge_indexp1,z, mu_neb, self.heads)
        return alpha,beta



class DiffGroupNorm1(torch.nn.Module):

    def __init__(self, in_channels, groups, lamda=0.01, eps=1e-5, momentum=0.1,
                affine=True, track_running_stats=True):
        super().__init__()

        self.in_channels = in_channels
        self.groups = groups
        self.lamda = lamda

        self.lin = nn.Linear(in_channels, groups, bias=False)
        self.norm = BatchNorm1d(groups * in_channels, eps, momentum, affine,
                                track_running_stats)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.norm.reset_parameters()


    def forward(self, x: Tensor) -> Tensor:
        """"""
        F, G = self.in_channels, self.groups

        s = self.lin(x).softmax(dim=-1)  # [N, G]
        out = s.unsqueeze(-1) * x.unsqueeze(-2)  # [N, G, F]
        out = self.norm(out.view(-1, G * F)).view(-1, G, F).sum(-2)  # [N, F]

        return x + self.lamda * out


    @staticmethod
    def group_distance_ratio(x: Tensor, y: Tensor, eps: float = 1e-5) -> float:
        r"""Measures the ratio of inter-group distance over intra-group
        distance

        .. math::
            R_{\text{Group}} = \frac{\frac{1}{(C-1)^2} \sum_{i!=j}
            \frac{1}{|\mathbf{X}_i||\mathbf{X}_j|} \sum_{\mathbf{x}_{iv}
            \in \mathbf{X}_i } \sum_{\mathbf{x}_{jv^{\prime}} \in \mathbf{X}_j}
            {\| \mathbf{x}_{iv} - \mathbf{x}_{jv^{\prime}} \|}_2 }{
            \frac{1}{C} \sum_{i} \frac{1}{{|\mathbf{X}_i|}^2}
            \sum_{\mathbf{x}_{iv}, \mathbf{x}_{iv^{\prime}} \in \mathbf{X}_i }
            {\| \mathbf{x}_{iv} - \mathbf{x}_{iv^{\prime}} \|}_2 }

        where :math:`\mathbf{X}_i` denotes the set of all nodes that belong to
        class :math:`i`, and :math:`C` denotes the total number of classes in
        :obj:`y`.
        """
        num_classes = int(y.max()) + 1

        numerator = 0.
        for i in range(num_classes):
            mask = y == i
            dist = torch.cdist(x[mask].unsqueeze(0), x[~mask].unsqueeze(0))
            numerator += (1 / dist.numel()) * float(dist.sum())
        numerator *= 1 / (num_classes - 1)**2

        denominator = 0.
        for i in range(num_classes):
            mask = y == i
            dist = torch.cdist(x[mask].unsqueeze(0), x[mask].unsqueeze(0))
            denominator += (1 / dist.numel()) * float(dist.sum())
        denominator *= 1 / num_classes

        return numerator / (denominator + eps)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'groups={self.groups})')
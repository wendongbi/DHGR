import torch
import torch_geometric
from torch_geometric.data import Data
import os.path as osp
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
from torch import Tensor
from torch_geometric.typing import OptPairTensor, Adj, Size
from tqdm import tqdm
from collections import Counter
import torch
import torch_geometric
from torch_geometric.data import Data
import os.path as osp
import os
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.nn as nn
# from torch_geometric.nn import SplineConv, GCNConv
from torch_sparse.mul import mul
import numpy as np
import time
import sys
import logging
from torch.nn.utils import clip_grad_norm_
import random
import seaborn as sns
from torch_sparse import coalesce, matmul
import copy
import torch_sparse
import copy
from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops
torch.autograd.set_detect_anomaly(True)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

# Similiar Encoder

class SimEncoder(nn.Module):
    def __init__(self, in_size, emb_size=64, num_classes=None, use_clf=False, \
            thres_min_deg=3., thres_min_deg_ratio=1.0, moment=1, use_cpu_cache=False, use_center_moment=True):
        super(SimEncoder, self).__init__()
        self.use_clf = use_clf
        self.moment = moment
        self.use_center_moment = use_center_moment
        self.thres_min_deg = thres_min_deg
        self.thres_min_deg_ratio = thres_min_deg_ratio
        self.use_cpu_cache = use_cpu_cache
        self.encoder_pre = nn.Sequential(
            nn.Linear(in_size, emb_size),
        )
        # self.encoder_pre = nn.Sequential(
        #     nn.Linear(in_size, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, emb_size),
        # )
        
        # self.encoder_post = nn.Sequential(
        #     nn.Linear(emb_size, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 128),
        # )
        self.encoder_post = nn.Sequential(
            nn.Linear(emb_size * moment, emb_size),
        )
        # self.encoder_post = nn.Sequential(
        #     nn.Linear(in_size, in_size),
        # )
        if use_clf:
            self.decoder = nn.Linear(emb_size, num_classes)

        self.x_sim = None
        self.adj_t_cache = None

        self.lbl_neb_mask = None
        self.lbl_sim = None

        self.reset_parameters()
    def reset_parameters(self):
        for m in self.encoder_pre:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.encoder_post:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        if self.use_clf:
            nn.init.kaiming_normal_(self.decoder.weight)
            nn.init.constant_(self.decoder.bias, 0)
    
    def moment_calculation(self, adj_t: SparseTensor,
                              x: Tensor,
                              moment: int = 3):
        mu = matmul(adj_t, x, reduce='mean')

        out = [mu]
        if moment > 1:
            if self.use_center_moment:
                sigma = matmul(adj_t, (x - mu).pow(2), reduce='mean')
            else:
                sigma = matmul(adj_t, (x).pow(2), reduce='mean')
            
            sigma[sigma == 0] = 1e-16
            sigma = sigma.sqrt()
            out.append(sigma)

            for order in range(3, moment+1):
                gamma = matmul(adj_t, x.pow(order), reduce='mean')
                mask_neg = None
                if torch.any(gamma == 0):
                    gamma[gamma == 0] = 1e-16
                if torch.any(gamma < 0):
                    mask_neg = gamma < 0
                    gamma[mask_neg] *= -1
                
                gamma = gamma.pow(1/order)
                if mask_neg != None:
                    gamma[mask_neg] *= -1
                out.append(gamma)
        # print(len(out), adj_t)
        return out

    def cal_similarity_batch(self, x, adj_t, batch_src, batch_tar, embedding=False, cat_self=False):
        # x = x - x.mean(dim=0, keepdim=True) # subtract center

        # x_dist = matmul(adj_t, x, reduce='sum')
        # x_dist = torch.cat(self.moment_calculation(adj_t, x_emb, moment=1), dim=-1)
        x_dist = torch.cat(self.moment_calculation(adj_t, x, moment=self.moment), dim=-1)
        if embedding:
            x_dist = self.encoder_post(x_dist)
        
        # concat self
        if cat_self:
            x_dist = torch.cat([x, x_dist], dim=-1)
        # Decentralize
        x_dist -= x_dist.detach().mean(dim=0, keepdim=True)

        x_dist_src = x_dist[batch_src]
        x_dist_tar = x_dist[batch_tar]
        # norm1 = x_dist_src.norm(p=2, dim=1).view(-1, 1)
        # norm2 = x_dist_tar.norm(p=2, dim=1).view(-1, 1)
        # norm = torch.matmul(norm1, norm2.T) + 1e-8
        # sim = torch.matmul(x_dist_src, x_dist_tar.T) / norm # n1, n2
        x_dist_src = F.normalize(x_dist_src, p=2, dim=-1, eps=1e-8)
        x_dist_tar = F.normalize(x_dist_tar, p=2, dim=-1, eps=1e-8)
        sim = torch.matmul(x_dist_src, x_dist_tar.T)

        # # dot product
        # sim = torch.matmul(x_dist, x_dist.T)
        return sim
    
    def cal_similarity(self, x, adj_t, embedding=False, cat_self=False, device=None):
        # x = x - x.mean(dim=0, keepdim=True) # subtract center

        # x_dist = matmul(adj_t, x, reduce='sum')
        # x_dist = torch.cat(self.moment_calculation(adj_t, x_emb, moment=self.moment), dim=-1)
        x_dist = torch.cat(self.moment_calculation(adj_t, x, moment=self.moment), dim=-1)
        if embedding:
            x_dist = self.encoder_post(x_dist)
        
        # concat self
        if cat_self:
            x_dist = torch.cat([x, x_dist], dim=-1)

        x_dist -= x_dist.detach().mean(dim=0, keepdim=True)
        
        norm = x_dist.norm(p=2, dim=1).view(-1, 1)
        norm = torch.matmul(norm, norm.T) + 1e-8
        sim = torch.matmul(x_dist, x_dist.T) / norm
        # x_dist = F.normalize(x_dist, p=2, dim=-1, eps=1e-8)
        # sim = torch.matmul(x_dist, x_dist.T)

        # # dot product
        # sim = torch.matmul(x_dist, x_dist.T)

        if device is not None:
            sim = sim.to(device)
        return sim
    
    def cal_emb_y_distance(self, emb_sim, y_sim, mask):
        # diff = (self.gather(emb_sim, mask, mask) - self.gather(y_sim, mask, mask)).pow(2)
        diff = (emb_sim - self.gather(y_sim, mask, mask, emb_sim.device)).pow(2)
        return diff
    
    def gather(self, mat, idx1, idx2, device=None):
        if device:
            mat = mat.to(device)
            idx1 = idx1.to(device)
            idx2 = idx2.to(device)
        mat = mat[idx1, :]
        mat = mat[:, idx2]
        
        return mat
    
    def forward(self, data, batch_src=None, batch_tar=None, finetune=False, use_landmark=True):
        x, edge_index = data.x, data.edge_index
        if  self.adj_t_cache == None:
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
        
        if self.lbl_sim is None or self.lbl_neb_mask is None:
            print('preparing label similarity matrix.')
            device = x.device
            self.adj_t_cache = self.adj_t_cache.cpu()
            self.prepare_lbl_sim(data.cpu(), thres_min_deg=self.thres_min_deg, thres_min_deg_ratio=self.thres_min_deg_ratio)
            data = data.to(device)
            self.adj_t_cache = self.adj_t_cache.to(device)
            self.lbl_neb_mask = self.lbl_neb_mask.to(device)
            
        if self.x_sim is None:
            print('preparing feature similarity matrix.')
            self.prepare_x_sim(x)
            print(self.x_sim.device, self.lbl_sim.device, self.lbl_neb_mask.device)
        
        if use_landmark and not finetune:
            lbl_neb_idxs = torch.where(self.lbl_neb_mask)[0].to(batch_src.device)
            batch_src = torch.cat([batch_src, lbl_neb_idxs], dim=-1).unique(sorted=False)
            batch_tar = torch.cat([batch_tar, lbl_neb_idxs], dim = -1).unique(sorted=False)

            
        
        emb = self.encoder_pre(x)
        # emb = x
        if self.use_clf:
            # classification head
            logits = self.decoder(F.dropout(emb, p=0.5, training=self.training))
            log_probs = F.log_softmax(logits, dim=-1)
            loss_clf = F.nll_loss(log_probs[data.train_mask], data.y[data.train_mask])
            accs =  []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = log_probs.detach()[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
            print(' loss_clf:{:.4f} | train:{:.3f} | val{:.3f} | test:{:.3f}'.format( loss_clf.detach().item(), *accs))

        # recons_loss = (emb_sim - self.x_sim).pow(2).mean().sqrt()
        if finetune:
            # emb_sim = self.cal_similarity(emb, self.adj_t_cache, embedding=True) # (n1,n2)
            emb_sim = self.cal_similarity_batch(emb, self.adj_t_cache, self.lbl_neb_mask, self.lbl_neb_mask, embedding=True) # (N,N)
            # emb_y_dist_diff = self.cal_emb_y_distance(emb_sim, self.lbl_sim, self.lbl_neb_mask)
            # emb_y_dist_diff = (self.gather(emb_sim, self.lbl_neb_mask, self.lbl_neb_mask) - self.gather(self.lbl_sim, self.lbl_neb_mask, self.lbl_neb_mask)).pow(2)
            # emb_y_dist_diff = (emb_sim - self.gather(self.lbl_sim, self.lbl_neb_mask, self.lbl_neb_mask, emb_sim.device)).pow(2)
            emb_y_dist_diff = (emb_sim - self.lbl_sim.to(x.device)).pow(2)
            loss = (emb_y_dist_diff.sum() / self.lbl_neb_mask.sum().float()).sqrt()
            # loss = (emb_y_dist_diff.sum() / self.lbl_neb_mask.sum().float().sqrt()).sqrt()
        else:
            if not self.use_cpu_cache:
                emb_sim = self.cal_similarity(emb, self.adj_t_cache, embedding=True) # (N,N)
                recons_loss = (self.gather(emb_sim, batch_src, batch_tar, x.device) - self.gather(self.x_sim, batch_src, batch_tar, x.device)).pow(2)
            else:
                emb_sim = self.cal_similarity_batch(emb, self.adj_t_cache, batch_src, batch_tar, embedding=True) # (n1,n2)
                recons_loss = (emb_sim - self.gather(self.x_sim, batch_src, batch_tar, x.device)).pow(2)
            loss = recons_loss.mean().sqrt()

            # emb_sim_lbl = self.cal_similarity_batch(emb, self.adj_t_cache, self.lbl_neb_mask, self.lbl_neb_mask, embedding=True) # (N,N)
            # emb_y_dist_diff = (emb_sim_lbl - self.gather(self.lbl_sim, self.lbl_neb_mask, self.lbl_neb_mask, emb_sim_lbl.device)).pow(2)
            # loss = recons_loss.mean().sqrt() + (emb_y_dist_diff.sum() / self.lbl_neb_mask.sum().float()).sqrt()
            # # loss = (recons_loss + emb_y_dist_diff).mean().sqrt()
        # return loss, recons_loss.mean().sqrt(), (emb_y_dist_diff.sum() / self.lbl_neb_mask.sum().float().sqrt()).sqrt()
        if self.use_clf:
            loss = loss + loss_clf
        return loss
        # return loss, recons_loss.mean().sqrt(), torch.zeros(1)

    def get_emb_sim(self, x, embedding=True, cat_self=False):
        emb = self.encoder_pre(x)
        # emb = x
        emb_sim = self.cal_similarity(emb, self.adj_t_cache, embedding=embedding, cat_self=cat_self)
        return emb_sim, emb
    
    def prepare_candidates(self, q=0.25):
        thres_pos = np.quantile(self.x_sim.view(-1).numpy(), q = 1 - q)
        thres_neg = np.quantile(self.x_sim.view(-1).numpy(), q = q)
        print('Preparing candidate set(thres_pos={}, thres_neg={})'.format(thres_pos, thres_neg))
    def prepare_x_sim(self, x):
        if self.use_cpu_cache:
            # the computation of x_sim is on cpu(slower but do not use gpu memory)
            x_dist = torch.cat(self.moment_calculation(self.adj_t_cache.cpu(), x.cpu(), moment=self.moment), dim=-1).cpu()
            x_dist -= x_dist.mean(dim=0, keepdim=True)
            x_dist = F.normalize(x_dist, p=2, dim=-1, eps=1e-8)
            self.x_sim = torch.matmul(x_dist, x_dist.T)
        else:
            # the computation of x_sim is on GPU(faster but using more gpu memory), but finally move x_sim to cpu
            self.x_sim = self.cal_similarity(x, self.adj_t_cache, device=torch.device('cpu'), embedding=False) # (N, N)
    def prepare_lbl_sim(self, data, thres_min_deg=3., thres_min_deg_ratio=1.8, max_cand_num=10000):
        N = data.num_nodes
        x, edge_index = data.x ,data.edge_index
        # lbl_sim
        edge_index = coalesce(edge_index, None, N, N)[0]
        _, col = edge_index
        mask_src_train = data.train_mask[data.edge_index[0]]
        deg_mask = degree(col[mask_src_train], x.size(0), dtype=x.dtype) # in degree, only count the src nodes in the training set.
        self.lbl_neb_mask = (deg_mask >= thres_min_deg)
        deg = degree(col, x.size(0), dtype=x.dtype)
        self.lbl_neb_mask_ratio = (deg_mask.float() / (1e-5 + deg.float())) > thres_min_deg_ratio
        self.lbl_neb_mask += self.lbl_neb_mask_ratio
        # self.lbl_neb_mask *= self.lbl_neb_mask_ratio
        if self.lbl_neb_mask.sum() > max_cand_num:
            thres_min_deg_old = thres_min_deg
            thres_min_deg = np.quantile(deg_mask.view(-1).numpy(), q = (1 - max_cand_num/N))
            print('thres_min_deg ajust from {} to {}'.format(thres_min_deg_old, thres_min_deg))
            self.lbl_neb_mask = (deg_mask >= thres_min_deg)
        print('candidate set size:', self.lbl_neb_mask.sum())
        # self.lbl_neb_mask = torch.matmul(self.lbl_neb_mask.float().unsqueeze(-1), self.lbl_neb_mask.float().unsqueeze(0)).bool() # (N, N)
        if data.y.min() == -1:
            # missing data
            mask_missing = (data.y == -1)
            data.y[mask_missing] = 0
            y_onehot = F.one_hot(data.y)
            data.y[mask_missing] = -1
            y_onehot[mask_missing, :] = 0
        else:
            y_onehot = F.one_hot(data.y)
        # x_lbl = y_onehot
        x_lbl = torch.zeros_like(y_onehot)
        x_lbl[data.train_mask] = y_onehot[data.train_mask]
        x_lbl = x_lbl.float().to(edge_index.device)

        # label distribution 1st-order
        lbl_dist = matmul(self.adj_t_cache, x_lbl, reduce='sum')[self.lbl_neb_mask]
        # norm = lbl_dist.norm(p=2, dim=1).view(-1, 1)
        # norm = torch.matmul(norm, norm.T) # (N, N)
        # norm[norm == 0] = 1e-8
        # lbl_sim = torch.matmul(lbl_dist, lbl_dist.T) / norm # (N, N)
        lbl_dist = F.normalize(lbl_dist, p=2, dim=-1)
        lbl_sim = torch.matmul(lbl_dist, lbl_dist.T)
        # print((lbl_sim > 0.99).sum(1).float().mean(), lbl_sim[0])
        
        # label distribution 2rd-order
        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)
        edge_index_2nd, value = torch_sparse.spspmm(edge_index, value, edge_index, value, N, N, N)
        value.fill_(0)
        edge_index_2nd, value = remove_self_loops(edge_index_2nd, value)
        adj_sp_2nd = torch_sparse.SparseTensor(row=edge_index_2nd[0], col=edge_index_2nd[1], value=torch.ones(edge_index_2nd.shape[1]).to(edge_index.device), sparse_sizes=(N, N))
        # lbl_dist_2nd = matmul(adj_sp_2rd.to(edge_index.device), x_lbl, reduce='sum')
        lbl_dist_2nd = matmul(adj_sp_2nd, x_lbl, reduce='sum')[self.lbl_neb_mask]
        # norm2 = lbl_dist_2nd.norm(p=2, dim=1).view(-1, 1)
        # norm2 = torch.matmul(norm2, norm2.T) + 1e-8
        # lbl_sim_2nd = torch.matmul(lbl_dist_2nd, lbl_dist_2nd.T) / norm2 # (N, N)
        lbl_dist_2nd = F.normalize(lbl_dist_2nd, p=2, dim=-1)
        lbl_sim_2nd = torch.matmul(lbl_dist_2nd, lbl_dist_2nd.T)
        # print((lbl_sim_2nd > 0.99).sum(1).float().mean(), lbl_sim_2nd[0])

        lbl_sim_merge = lbl_sim * lbl_sim_2nd # (N, N)
        # lbl_sim_merge = lbl_sim
        # print((lbl_sim_merge > 0.99).sum(1).float().mean(), lbl_sim_merge[0])
        self.lbl_sim = lbl_sim_merge
        # self.lbl_sim -= torch.diag_embed(self.lbl_sim.diag())

class ModelHandler(nn.Module):
    def __init__(self, in_size, num_classes, thres_min_deg=10., thres_min_deg_ratio=0.8, hidden=128, device=None, \
                save_dir='../ckpt/', seed=0, num_epoch=200, num_epoch_finetune=30, window_size=[10000, 10000], \
                lr=0.001, weight_decay=5e-3, shuffle=[False, False], drop_last=False, moment=1, use_cpu_cache=False):
        super(ModelHandler, self).__init__()
        assert device is not None
        self.device = device
        self.in_size = in_size
        self.num_classes = num_classes
        self.hidden = hidden
        self.save_dir = save_dir
        self.seed = seed
        # hyperparameters for training model
        self.trained = False # flag reflecting whether the model has already been trained
        self.num_epoch = num_epoch
        self.num_epoch_finetune = num_epoch_finetune
        self.window_size = window_size
        self.lr = lr
        self.wd = weight_decay
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_cpu_cache = use_cpu_cache
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # model parameters initialization
        set_random_seed(seed)
        self.graph_learner = SimEncoder(in_size=in_size, thres_min_deg=thres_min_deg, \
                    thres_min_deg_ratio=thres_min_deg_ratio, emb_size=hidden, \
                    num_classes=num_classes, use_clf=False, moment=moment, use_cpu_cache=use_cpu_cache)
        self.graph_learner = self.graph_learner.to(device)
    def train(self, data, num_epoch=200, num_epoch_finetune=30, \
              window_size=[10000, 10000], lr=0.001, weight_decay=5e-3, \
              shuffle=[False, False], drop_last=False):
        if self.trained:
            print('The model has already been trained')
            return
        if data.x.device != self.device:
            data = data.to(self.device)
        idxs = torch.arange(data.num_nodes)
        for i in range(len(window_size)):
            if window_size[i] == -1:
                window_size[i] = data.x.shape[0]
        src_idx_loader = torch.utils.data.DataLoader(idxs, batch_size=window_size[0], shuffle=shuffle[0], drop_last=drop_last[0])
        tar_idx_loader = torch.utils.data.DataLoader(idxs, batch_size=window_size[1], shuffle=shuffle[1], drop_last=drop_last[1])

        set_random_seed(self.seed)
        optimizer = torch.optim.Adam(self.graph_learner.parameters(), lr=lr, weight_decay=weight_decay)
        save_path = os.path.join(self.save_dir, 'best_sim_encoder.ckpt')
        min_recons_loss = 10
        best_epoch = None
        self.graph_learner.train()
        for epoch in range(1, 1+num_epoch):
            t0 = time.time()
            loss_bucket = []
            for batch_src in src_idx_loader:
                for batch_tar in tar_idx_loader:
                    batch_src = batch_src.to(self.device)
                    batch_tar = batch_tar.to(self.device)
                    loss_train = self.train_Sim_batch(data, self.graph_learner, optimizer, batch_src, batch_tar)
                    loss_bucket.append(loss_train)
            loss = np.mean(loss_bucket)
            if loss < min_recons_loss:
                min_recons_loss = loss
                torch.save(self.graph_learner.state_dict(), save_path)
                best_epoch = epoch
            log = 'Epoch: {:03d}, Reconstruction Loss:{:.4f} Time(s/epoch):{:.4f}'.format(epoch, loss, time.time() - t0)
            print(log)
        print('[Train] best epoch:{} | min reconstruction loss:{:.4f}'.format(best_epoch, min_recons_loss))
        # print('Loading best parameters from {}'.format(save_path))
        # print(self.graph_learner.load_state_dict(torch.load(save_path)))
        print('Begin finetuning...')
        min_sup_loss = 100
        for epoch in range(1, 1+num_epoch_finetune):
            t0 = time.time()
            loss_finetune = self.finetune_Sim(data, self.graph_learner, optimizer)
            # if loss_finetune < min_sup_loss:
            #     best_epoch = epoch
            #     min_sup_loss = loss_finetune
            #     torch.save(self.graph_learner.state_dict(), save_path)
            log = 'Epoch: {:03d}, Supervised Loss:{:.4f} Time(s/epoch):{:.4f}'.format(epoch, loss_finetune, time.time() - t0)
            print(log)
        # print('[Finetune] best epoch:{} | min supervised loss:{:.4f}\n Loading best params...'.format(best_epoch, min_sup_loss))
        # print(self.graph_learner.load_state_dict(torch.load(save_path)))
        self.trained = True # make sure that the model can only be trained once
    def train_Sim_batch(self, data, model, optimizer, batch_src, batch_tar, clip_grad=False):
        model.train()
        optimizer.zero_grad()
        loss = model(data, batch_src, batch_tar)
        loss.backward()

        if clip_grad:
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach().item()
    def finetune_Sim(self, data, model, optimizer, clip_grad=False):
        model.train()
        optimizer.zero_grad()
        loss = model(data, finetune=True)
        # print('emb_y_dist_diff:{:.4f}'.format( loss.detach().item()))
        loss.backward()
        if clip_grad:
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach().item()
    def get_emb(self, data, embedding=True, cat_self=True):
        with torch.no_grad():
            self.graph_learner.eval()
            emb_sim, emb = self.graph_learner.get_emb_sim(data.x, embedding=embedding, cat_self=cat_self)
        return emb_sim, emb
    def knn_epsilon_graph(self, data, k=8, thres_lower_sim=None, embedding=True, cat_self=True):
        # feat_sim = model.x_sim.cpu()
        feat_sim = self.get_emb(data, embedding=embedding, cat_self=cat_self)[0].cpu()
        # feat_sim = lbl_sim
        feat_sim -= torch.diag_embed(feat_sim.diag())
        if thres_lower_sim is None:
            thres_lower_sim = np.quantile(feat_sim.view(-1).numpy(), q=1 - 1.1 * data.num_edges / feat_sim.view(-1).shape[0])
        print(thres_lower_sim, data.num_edges)
        topk_lbl_sim = feat_sim.topk(k=k, dim=1, largest=True, sorted=False)
        row = torch.stack([torch.arange(0, data.x.shape[0]) for _ in range(k)], dim=-1).view(-1).to(self.device)
        col = topk_lbl_sim.indices.view(-1).to(self.device)
        assert row.shape == col.shape
        edge_index_topk_lbl_dist = torch.stack((row, col), dim=0).long()
        edge_index_topk_lbl_dist = edge_index_topk_lbl_dist[:, topk_lbl_sim.values.view(-1) >= thres_lower_sim]
        print('Add {} edges.'.format(edge_index_topk_lbl_dist.shape[1]))
        return edge_index_topk_lbl_dist
    def graph_prunning(self, data, thres_prunning=0., embedding=True, cat_self=True):
        # prunning on the original graph
        edge_index = data.edge_index
        
        feat_sim = self.get_emb(data, embedding=embedding, cat_self=cat_self)[0]#.cpu()
        mask_prunning = feat_sim[edge_index[0], edge_index[1]]
        # thres_quantile = 0.5
        # print('thres:', np.quantile(mask_prunning.cpu().numpy(), q=thres_quantile), mask_prunning.mean())
        # mask_prunning = mask_prunning > np.quantile(mask_prunning.cpu().numpy(), q=thres_quantile)
        mask_prunning = mask_prunning > thres_prunning
        # print(mask_prunning.sum() / mask_prunning.shape[0], '#' * 20)
        edge_index_prunning = edge_index[:, mask_prunning]
        print('Prune {} edges from {} to {}'.format(mask_prunning.shape[0] - mask_prunning.sum(), edge_index.shape, edge_index_prunning.shape))
        return edge_index_prunning
    def merge_graph(self, data, edge_index_topk_lbl_dist, edge_index_prunning=None):
        # replace with label-distributional-graph
        data_modi = copy.deepcopy(data).to(self.device)
        # data_modi.edge_index = edge_index_topk_lbl_dist.to(device)
        # data_modi.edge_index = torch.stack((torch.arange(x.shape[0]), torch.arange(x.shape[0])), dim=0).to(device)
        
        if edge_index_prunning is not None:
            edge_index_merge = coalesce(torch.cat((edge_index_prunning, edge_index_topk_lbl_dist), dim=-1), None, data.num_nodes, data.num_nodes)[0]
        else:
            edge_index_merge = coalesce(torch.cat((data.edge_index, edge_index_topk_lbl_dist), dim=-1), None, data.num_nodes, data.num_nodes)[0]
        # edge_index_merge = edge_index_prunning
        # edge_index_merge = edge_index_lbl.to(device)
        # edge_index_merge = coalesce(torch.cat((edge_index, edge_index_lbl), dim=-1), None, data.num_nodes, data.num_nodes)[0]
        data_modi.edge_index = edge_index_merge.to(self.device)
        return data_modi
    def forward(self, data, k=8, epsilon=None, embedding_post=True, cat_self=True, prunning=False, thres_prunning=0., load_path=None, save_path=None):
        # load existing graph
        if load_path is not None:
            print('load graph from:', load_path)
            return torch.load(load_path).to(data.x.device)
        # train graph learner
        self.train(data, num_epoch=self.num_epoch, num_epoch_finetune=self.num_epoch_finetune, \
              window_size=self.window_size, lr=self.lr, weight_decay=self.wd, \
              shuffle=self.shuffle, drop_last=self.drop_last)
        if self.use_cpu_cache:
            cpu_device = torch.device('cpu')
            data = data.to(cpu_device)
            self.graph_learner = self.graph_learner.to(cpu_device)
            self.graph_learner.adj_t_cache = self.graph_learner.adj_t_cache.to(cpu_device)
        # add edges
        edge_index_topk_lbl_dist = self.knn_epsilon_graph(data, k=k, thres_lower_sim=epsilon, embedding=embedding_post, cat_self=cat_self)
        # del edges
        edge_index_prunning = None
        # print('debug:', data.x.device)
        if prunning:
            edge_index_prunning = self.graph_prunning(data, thres_prunning=thres_prunning)
        # get final graph
        if self.use_cpu_cache:
            data = data.to(self.device)
            edge_index_topk_lbl_dist = edge_index_topk_lbl_dist.to(self.device) 
            edge_index_prunning = edge_index_prunning.to(self.device) if prunning else None
        data_new = self.merge_graph(data, edge_index_topk_lbl_dist, edge_index_prunning)
        if save_path is not None:
            print('save graph to:', save_path)
            torch.save(data_new.to(torch.device('cpu')), save_path)
        return data_new.to(data.x.device)


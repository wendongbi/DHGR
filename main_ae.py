import torch
import torch_geometric
from torch_geometric.data import Data
import os.path as osp
import os
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull, AttributedGraphDataset, Reddit, AttributedGraphDataset,  WebKB, WikipediaNetwork, Actor, Flickr
from torch_geometric.nn.conv import sage_conv
import torch_geometric.transforms as T
import torch.nn as nn
# from torch_geometric.nn import SplineConv, GCNConv
from torch_geometric.nn import SplineConv, GATConv, SAGEConv
from torch_sparse.mul import mul
print(torch_geometric.__version__)
import numpy as np
import time
import sys
import logging
from torch.nn.utils import clip_grad_norm_
sys.path.append('./')
sys.path.append('./models')
import model
from dataset import Facebook100, prepare_dataset_benchmark, Facebook100_heterphily
import random
import matplotlib.pyplot as plt
# from model import HM_GCNNet, HM_SAGENet
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from GraphLearner import ModelHandler
torch.autograd.set_detect_anomaly(True)

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



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def col_norm(x, mode='std'):
    assert isinstance(x, torch.Tensor)
    if mode == 'std':
        mu = x.mean(dim=0)
        sigma = x.std(dim=0)
        out = (x - mu) / sigma
    return out

    

def train(data, model, optimizer, clip_grad=False):
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
    loss.backward()
    if clip_grad:
        clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach().item()


# @torch.no_grad()
def test(data, model, dataset_name):
    with torch.no_grad():
        model.eval()
        log_probs, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = log_probs[mask].max(1)[1]
            score = pred.eq(data.y[mask]).sum().item() / mask.sum().item() # acc
            # score = f1_score(data.y[mask].cpu().numpy(), pred.detach().cpu().numpy(), average='micro')
            accs.append(score)
        return accs


def train_AE(data, model, optimizer, clip_grad=False):
    model.train()
    optimizer.zero_grad()

    # out = model(data)
    out, loss_recons, Mahalanobis_dist_list = model(data)
    loss_clf = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss = loss_clf + loss_recons + sum(Mahalanobis_dist_list)
    # loss = loss_clf + loss_recons
    print('Mahalanobis_dist: {:.4f}'.format(sum(Mahalanobis_dist_list).detach().item()))
    loss.backward()
    if clip_grad:
        clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2, error_if_nonfinite=True)
    optimizer.step()
    return loss_clf.detach().item(), loss_recons.detach().item()

# @torch.no_grad()
def test_AE(data, model, dataset_name):
    with torch.no_grad():
        model.eval()
        log_probs, _, _ = model(data)
        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = log_probs[mask].max(1)[1]
            score = pred.eq(data.y[mask]).sum().item() / mask.sum().item() # acc
            # score = f1_score(data.y[mask].cpu().numpy(), pred.detach().cpu().numpy(), average='micro')
            accs.append(score)
        return accs

def build_model(args, dataset, device: torch.device, model_init_seed=None, edge_index=None):
    if model_init_seed is not None:
        set_random_seed(model_init_seed)
    if args.model == 'ConvGNN':
        from model import ConvNet
        model = ConvNet(dataset)
    elif args.model == 'GCN':
        from model import GCNNet
        model = GCNNet(dataset, args.num_layer, hidden=args.hidden)
    elif args.model == 'GAT':
        from model import GAT
        model = GAT(dataset=dataset, hidden=args.hidden)
    elif args.model == 'GIN':
        from model import GINNet
        model = GINNet(dataset, args.num_layer, hidden=args.hidden)
    elif args.model == 'APPNP':
        from model import APPNP_Net
        model = APPNP_Net(dataset=dataset, hidden=args.hidden)
    elif args.model == 'GCN2':
        from model import GCN2
        model = GCN2(dataset, hidden_channels=args.hidden, num_layers=args.num_layer, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6)
    elif args.model == 'JKNet':
        from model import JKNet
        model = JKNet(in_channels=dataset.num_features, hidden_channels=args.hidden, out_channels=dataset.num_classes, num_layers=args.num_layer, dropout=0.5, mode='cat')
    elif args.model == 'DeeperGCN':
        from model import DeeperGCN
        model = DeeperGCN(dataset, hidden_channels=args.hidden, num_layers=args.num_layer) # num_layers=28
    elif args.model == 'GraphSAGE':
        from model import GraphSAGE
        model = GraphSAGE(dataset, args.num_layer, hidden=args.hidden)
    elif args.model == 'DistGNN':
        from Moment_Metric_GNN import myGATv2
        model = myGATv2(in_channels=dataset.num_features, hidden_channels=args.hidden, out_channels=dataset.num_classes, num_layers=args.num_layer, heads=1, dropout=0.5, att_dropout=0.5)
    elif args.model == 'DistGNN_AE':
        from myGATv2_AE import myGATv2
        model = myGATv2(in_channels=dataset.num_features, hidden_channels=args.hidden, out_channels=dataset.num_classes, num_layers=args.num_layer, heads=1, dropout=0.5, att_dropout=0.5)
    elif args.model == 'GPRGNN':
        from models_benchmark import GPRGNN
        model = GPRGNN(dataset.num_features, args.hidden, dataset.num_classes)
    elif args.model == 'MixHop':
        from models_benchmark import MixHop
        model = MixHop(dataset.num_features, args.hidden, dataset.num_classes)
    elif args.model == 'H2GCN':
        from models_benchmark import H2GCN
        assert edge_index is not None
        N = dataset.num_nodes if args.dataset == 'fb100' else dataset[0].num_nodes
        model = H2GCN(dataset.num_features, args.hidden, dataset.num_classes, edge_index.to(device), N)
    else:
        raise NotImplementedError('Not implemented model: {}'.format(args.model))
    return model.to(device)

def build_dataset(args, transform=None):
    data_split = None
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        path = osp.join(args.data_dir, args.dataset)
        dataset = Planetoid(path, args.dataset, transform=transform, split='public')
    elif args.dataset == 'Reddit':
        path = osp.join(args.data_dir, args.dataset)
        if transform != None:
            dataset = Reddit(path, transform=transform)
        else:
            dataset = Reddit(path)
    elif args.dataset == 'Flickr':
        path = osp.join(args.data_dir, args.dataset)
        dataset = Flickr(path, transform=transform)
    elif args.dataset == 'BlogCatalog':
        path = osp.join(args.data_dir, args.dataset)
        if transform != None:
            dataset = AttributedGraphDataset(path, args.dataset, transform=transform)
        else:
            dataset = AttributedGraphDataset(path, args.dataset)
        # data_split = torch.load(osp.join(path, 'data_split.bin'))
    elif args.dataset in ["Cornell", "Texas", "Wisconsin"]:
        path = osp.join(args.data_dir, args.dataset)
        if transform != None:
            dataset = WebKB(path, args.dataset, transform=transform)
        else:
            dataset = WebKB(path, args.dataset)
    elif args.dataset == 'Actor':
        path = osp.join(args.data_dir, args.dataset)
        if transform != None:
            dataset = Actor(path, transform=transform)
        else:
            dataset = Actor(path)
    elif args.dataset in ["chameleon", "squirrel"]:
        path = osp.join(args.data_dir, args.dataset)
        dataset = WikipediaNetwork(path, args.dataset)
    elif args.dataset in ['fb100', 'arxiv-year']:
        sub_dataname = 'Penn94' if args.dataset == 'fb100' else ''
        dataset = prepare_dataset_benchmark(args.dataset, sub_dataname=sub_dataname)
    elif args.dataset in ['Penn94']:
        path = osp.join(args.data_dir, args.dataset)
        dataset = Facebook100_heterphily(path, args.dataset, transform=transform, split='random', \
            num_val=500, num_test=None, num_train_per_class=200, to_onehot=True, train_val_test_ratio=[0.6,0.2,0.2])
    else:
        path = osp.join(args.data_dir, args.dataset)
        dataset = Facebook100(path, args.dataset, transform=transform, split='random', \
            num_val=500, num_test=None, num_train_per_class=200, to_onehot=True, train_val_test_ratio=[0.6,0.2,0.2])
    return dataset, data_split

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Training scripts for Higher Moment GNN model')
    ap.add_argument('--model', type=str, help='model name')
    ap.add_argument('--num_layer', type=int, default=2)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--num_epoch', type=int, default=200)
    ap.add_argument('--dataset', type=str, default='Cora',  help='dataset name')
    ap.add_argument('--data_dir', type=str, default='./data', help='path of dir of datasets')
    ap.add_argument('--gpu', type=str, default='0', help='id of gpu card to use')
    ap.add_argument('--running_id', type=str, default='0', help='experiment id for logging output')
    ap.add_argument('--log_dir', type=str, default=None, help='dir of log files, do not log if None')
    ap.add_argument('--hidden', type=int, default=64, help='fixed random seed by torch.manual_seed')
    ap.add_argument('--moment', type=int, default=1, help='max moment used in multi-moment graphSAGE model(MM_SAGE)')
    ap.add_argument('--seed', type=int, default=0, help='fixed random seed by torch.manual_seed')
    ap.add_argument('--gnn_seed', type=str, default=None, help='fixed random seed by torch.manual_seed, for example [6,66,666]')
    ap.add_argument('--auto_fixed_seed', action='store_true', help='fixed random seed of each run by run_id(0, 1, 2, ...)')
    ap.add_argument('--use_center_moment', action='store_true', help='whether to use center moment for MM_SAGE')
    ap.add_argument('--lr', type=float, default=0.01, help='learning rate')
    ap.add_argument('--wd', type=float, default=5e-3, help='weight decay')
    
    ap.add_argument('--graph_learn', action='store_true', help='whether to use graph_learn')
    ap.add_argument('--lr_gl', type=float, default=0.001, help='learning rate')
    ap.add_argument('--wd_gl', type=float, default=5e-3, help='weight decay')
    ap.add_argument('--thres_min_deg', type=int, default=3, help='threshold for minimum degree')
    ap.add_argument('--thres_min_deg_ratio', type=float, default=1.0, help='threshold for minimum degree ratio(train set in all)')
    ap.add_argument('--save_dir_gl', type=str, default='../ckpt/', help='dir of model params files for graph learner')
    ap.add_argument('--window', type=str, default='[10000, 10000]', help='window size for scalable training mode')
    ap.add_argument('--shuffle', type=str, default='[False, False]', help='whether to shuffle the batch data')
    ap.add_argument('--epoch_train_gl', type=int, default=200)
    ap.add_argument('--epoch_finetune_gl', type=int, default=30)
    ap.add_argument('--seed_gl', type=int, default=0)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--cat_self', type=str, default='True')
    ap.add_argument('--drop_last', type=str, default='[False,False]')
    ap.add_argument('--prunning', type=str, default='False')
    ap.add_argument('--epsilon', type=float, default=None)
    ap.add_argument('--thres_prunning', type=float, default=0.)
    ap.add_argument('--use_cpu_cache', action='store_true', default=False, help='whether to use cpu to cache the initial feat/label similarity for saving gpu memory.')
    # for ablation study
    ap.add_argument('--drop_edge', action='store_true', default=False, help='whether to random drop edges with p')
    ap.add_argument('--prob_drop_edge', type=float, default=0., help='prob of drop edge.')
    

    args = ap.parse_args()
    args.window = eval(args.window.replace(' ', ''))
    args.shuffle = eval(args.shuffle.replace(' ', ''))
    args.cat_self = eval(args.cat_self)
    args.prunning = eval(args.prunning)
    args.drop_last = eval(args.drop_last.replace(' ', ''))
    args.gnn_seed = eval(args.gnn_seed.replace(' ', '')) if args.gnn_seed is not None else None
    print(args)
    

    logger = None
    if args.log_dir is not None:
        if not osp.exists(args.log_dir):
            os.makedirs(args.log_dir)
        save_path = osp.join(args.log_dir, f'{args.dataset}_{args.model}_{args.num_layer}_{args.repeat}_{args.running_id}.log')
        # logger = open(save_path, 'w')
        # logging.basicConfig(level = logging.INFO,format = '[%(asctime)s]: %(message)s', filename=save_path)
        # logger = logging.getLogger(__name__)
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(save_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s]: %(message)s')
        handler.setFormatter(formatter)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console)
    else:
        logging.basicConfig(level = logging.INFO,format = '[%(asctime)s]: %(message)s')
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(device)
    

    final_acc = {
        'train': [],
        'val': [],
        'test': []
    }
    log_results = False
    model_handler = None
    split_idx_pre = None
    repeat = min(args.repeat, len(args.gnn_seed)) if args.gnn_seed is not None else args.repeat
    # save_path_gl = f'../ckpt/data_{args.dataset}_new.dat'
    save_path_gl = f'../ckpt/data_{args.dataset}_new_{args.thres_prunning}.dat'
    load_path_gl = None
    # load_path_gl = f'../ckpt/data_{args.dataset}_new_{args.thres_prunning}.dat'
    # load_path_gl = f'../ckpt/data_{args.dataset}_new.dat'

    runs = 1
    if args.dataset in ["Cornell", "Texas", "Wisconsin", "actor", "chameleon", "squirrel"]:
        runs = 10
    for split_idx in range(runs):
        for train_id in range(1, 1+repeat):
            logger.info('repeat {}/{}'.format(train_id, args.repeat))
            # split_idx = int((train_id - 1) / 3)
            # split_idx = 0
            # split_idx = args.seed
            # for split_idx in range(10):
            if args.gnn_seed is not None:
                data_split_seed = args.seed
                model_init_seed = args.gnn_seed[train_id-1]
                set_random_seed(data_split_seed)
                logger.info('auto fixed data split seed to {}, model init seed to {}'.format(data_split_seed, model_init_seed))
            elif args.seed is not None:
                logger.info('Manual random seed:{}'.format(args.seed))
                # torch.manual_seed(args.seed)
                data_split_seed = args.seed
                model_init_seed = train_id - 1
                # model_init_seed = args.seed
                # model_init_seed = (2 ** model_init_seed) * model_init_seed
                set_random_seed(data_split_seed)
                logger.info('auto fixed data split seed to {}, model init seed to {}'.format(data_split_seed, model_init_seed))
            elif args.auto_fixed_seed:
                logger.info('auto fixed random seed to {}'.format(train_id-1))
                data_split_seed = int((train_id - 1) / 3)
                model_init_seed = int((train_id - 1) % 3)
                set_random_seed(data_split_seed)
                logger.info('auto fixed data split seed to {}, model init seed to {}'.format(data_split_seed, model_init_seed))

            # build datasets
            transform = T.Compose([
                T.RandomNodeSplit(num_val=2000, num_test=2000),
                # T.TargetIndegree(),
            ])
            if args.dataset != 'BlogCatalog':
                transform = None
            dataset, data_split = build_dataset(args, transform=transform)
            # data_split=None
            if args.dataset in ["arxiv-year", "fb100"]:
                dataset.num_classes = 1 + dataset.y.max().item()
                data = dataset
            else:
                data = dataset[0]
            # if args.dataset in ['Amherst', 'Hamilton', 'Georgetown']:
            #     # data.x = col_norm(data.x, mode='std')
            #     data.x = onehot_encoder(data.x)

            
            # # save this data split
            # data_split = {
            #     'train': data.train_mask,
            #     'val': data.val_mask,
            #     'test': data.test_mask
            # }
            # path = os.path.join(args.data_dir, args.dataset)
            # torch.save(data_split, os.path.join(path, 'data_split.bin'))

            if args.dataset in ["Cornell", "Texas", "Wisconsin", "Actor", "chameleon", "squirrel", 'fb100']:
                data.train_mask = data.train_mask[:, split_idx]
                data.val_mask = data.val_mask[:, split_idx]
                data.test_mask = data.test_mask[:, split_idx]
            data = data.to(device)
            if data_split != None:
                data.train_mask = data_split['train'].to(device)
                data.val_mask = data_split['val'].to(device)
                data.test_mask = data_split['test'].to(device)
            t_start = time.time()
            if args.graph_learn:
                print('[Old Data]', data)
                hr_old = (data.y[data.edge_index[0]] == data.y[data.edge_index[1]]).sum() / data.edge_index.shape[1]
                if split_idx_pre is None or (split_idx_pre != None and split_idx_pre != split_idx):
                    # model_handler = ModelHandler(data.num_features, dataset.num_classes, thres_min_deg=3., thres_min_deg_ratio=1.8, hidden=128, device=device, \
                    #         save_dir='../ckpt/', seed=0, num_epoch=200, num_epoch_finetune=30, window_size=[10000, 10000], \
                    #         lr=0.001, weight_decay=5e-3, shuffle=[False, False], drop_last=False)
                    model_handler = ModelHandler(data.num_features, dataset.num_classes, thres_min_deg=args.thres_min_deg, thres_min_deg_ratio=args.thres_min_deg_ratio, hidden=128, device=device, \
                            save_dir=args.save_dir_gl, seed=args.seed_gl, num_epoch=args.epoch_train_gl, num_epoch_finetune=args.epoch_finetune_gl, window_size=args.window, \
                            lr=args.lr_gl, weight_decay=args.wd_gl, shuffle=args.shuffle, drop_last=args.drop_last, moment=args.moment, use_cpu_cache=args.use_cpu_cache)
                    split_idx_pre = split_idx
                    load_path_gl = None
                data = model_handler(data, k=args.k, epsilon=args.epsilon, embedding_post=True, cat_self=args.cat_self, prunning=args.prunning, thres_prunning=args.thres_prunning, load_path=load_path_gl, save_path=save_path_gl)
                load_path_gl = save_path_gl
                hr_new = (data.y[data.edge_index[0]] == data.y[data.edge_index[1]]).sum() / data.edge_index.shape[1]
                print('Homophily ratio change from {:.3f} to {:.3f}'.format(hr_old, hr_new))
            print(data)
            logger.info('[Dataset-{}] train_num:{}, val_num:{}, test_num:{}, class_num:{}'.format(args.dataset, data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item(), dataset.num_classes))


            # build model
            model = build_model(args, dataset, device, model_init_seed, data.edge_index)
            print(model)
            # build optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

            # drop edge
            if args.drop_edge:
                assert not args.graph_learn
                mask_edge = torch.ones(data.edge_index.shape[1])
                mask_edge = F.dropout(mask_edge, args.prob_drop_edge).bool()
                data.edge_index = data.edge_index[:, mask_edge]
                print('Random drop edges with probabilty {:.2f}'.format(args.prob_drop_edge))
                print(data)

            best_acc = {
                'train': 0,
                'val': 0,
                'test': 0
            }
            min_loss = 100.

            # save_attn_fig = seed_run == 0
            save_attn_fig = False
            if save_attn_fig:
                atten_bucket_layer1 = [[] for _ in range(args.moment)]
                atten_bucket_layer2 = [[] for _ in range(args.moment)]
            
            
            for epoch in range(1, 1+args.num_epoch):
                t0 = time.time()
                # record attention value of mode attn-1 or attn-2
                if save_attn_fig:
                    attn_score = F.softmax(model.convs[0].attn, dim=0)
                    # attn_score = attn_score[:, :, :, 1].mean(-1).mean(-1).detach().cpu().numpy()
                    attn_score = attn_score.mean(-1).mean(-1).mean(-1).detach().cpu().numpy()
                    attn_score = np.around(attn_score, 3)
                    for m in range(args.moment):
                        atten_bucket_layer1[m].append(attn_score[m])
                    attn_score = F.softmax(model.convs[1].attn, dim=0)
                    # attn_score = attn_score[:, :, :, 1].mean(-1).mean(-1).detach().cpu().numpy()
                    attn_score = attn_score.mean(-1).mean(-1).mean(-1).detach().cpu().numpy()
                    attn_score = np.around(attn_score, 3)
                    for m in range(args.moment):
                        atten_bucket_layer2[m].append(attn_score[m])
                    
                if args.model == 'DistGNN_AE':
                    loss_train, loss_recons = train_AE(data, model, optimizer)
                    eval_res = test_AE(data, model, dataset_name=args.dataset)
                    log = 'Epoch: {:03d}, Loss:{:.4f} Loss_clf:{:.4f} Loss_recons:{:.4f} Train: {:.4f}, Val:{:.4f}, Test: {:.4f}, Time(s/epoch):{:.4f}'.format(epoch, loss_train + loss_recons, loss_train, loss_recons, *eval_res, time.time() - t0)
                else:
                    loss_train = train(data, model, optimizer)
                    eval_res = test(data, model, dataset_name=args.dataset)
                    log = 'Epoch: {:03d}, Loss:{:.4f} Train: {:.4f}, Val:{:.4f}, Test: {:.4f}, Time(s/epoch):{:.4f}'.format(epoch, loss_train, *eval_res, time.time() - t0)
                
                
                
                logger.info(log)
                if eval_res[1] > best_acc['val']:
                # if eval_res[1] > best_acc['val'] and loss_train < min_loss:
                    min_loss = loss_train
                    best_acc['train'] = eval_res[0]
                    best_acc['val'] = eval_res[1]
                    best_acc['test'] = eval_res[2]
            print('Epoch running time:{:.4f} s'.format(time.time() - t_start))
            logger.info('[Run-{} score] {}'.format(train_id, best_acc))
            final_acc['train'].append(best_acc['train'])
            final_acc['val'].append(best_acc['val'])
            final_acc['test'].append(best_acc['test'])
        
    best_test_run  = np.argmax(final_acc['test'])
    final_acc_avg = {}
    final_acc_std = {}
    for key in final_acc:
        best_acc[key] = max(final_acc[key])
        final_acc_avg[key] = np.mean(final_acc[key])
        final_acc_std[key] = np.std(final_acc[key])
    logger.info('[Average Score] {} '.format(final_acc_avg))
    logger.info('[std Score] {} '.format(final_acc_std))
    logger.info('[Best Score] {}'.format(best_acc))
    logger.info('[Best test run] {}'.format(best_test_run))
    if log_results:
        f_path = './res/exp_log.csv'
        with open(f_path, 'a+') as fw:
            # fw.write('Dataset,Model,use_adj_norm,use_center_moment,moment,mode,avg_test_acc,exp_acc_list\n')
            avg_test_acc = final_acc_avg['test']
            test_acc_list = str(final_acc['test']).replace(',', ';').replace('[', '').replace(']', '').replace(' ', '')
            fw.write(f'{args.dataset},{args.model},{args.use_adj_norm},{args.use_center_moment},{args.moment},{args.mode},{avg_test_acc},{test_acc_list}\n')
    print(args)
    




from __future__ import division
from __future__ import print_function
import optuna
import random

import time
import argparse
import numpy as np
from copy import deepcopy as dcp
import scipy.sparse as sp
import math

import os.path as osp
import os
from torch_geometric.utils import dropout_adj, to_dense_adj, to_scipy_sparse_matrix, add_self_loops, dense_to_sparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, sparse_mx_to_torch_sparse_tensor, normalize, label_propagation, drop_feature, aug_random_mask, adj_nor
from models import GCN,MLP

from torch_geometric.datasets import Planetoid, CitationFull,WikiCS, Coauthor, Amazon
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--data_aug', type=int, default=1,
                    help='do data augmentation.')
parser.add_argument('--att_type', type=int, default=1,
                    help='0无neg,1neg选择')
parser.add_argument('--kk', type=int, default=1,
                    help='y_pre select k')
parser.add_argument('--cl_num', type=int, default=2,
                    help='0 train set;1 all dataset.')
parser.add_argument('--sample_size', type=float, default=0.,
                    help='sample size')
parser.add_argument('--encoder_type', type=int, default=3,
                    help='do data augmentation.')
parser.add_argument('--debias', type=float, default=0.,
                    help='debias rate.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--train_type', type=int, default=0,
                    help='0train ,1dataset')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight', type=float, default=0.,
                    help='Initial loss rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--tau', type=float, default=0.4,
                    help='tau rate .')
parser.add_argument('--dataset', type=str, default='Cora',
                    help='Cora/CiteSeer/PubMed/')
parser.add_argument('--encoder', type=str, default='GCN',
                    help='GCN/SGC/GAT/')
args = parser.parse_args()
times = 10

#load dataset
#def get_dataset(path, name):
#    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP','WikiCS','Amazon-Photo']
#    name = 'dblp' if name == 'DBLP' else name
#    print(name)
#    return (CitationFull if name == 'dblp' else Planetoid)(path,name,transform=T.NormalizeFeatures())
    
def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())

if args.dataset=='Cora' or args.dataset=='CiteSeer' or args.dataset=='PubMed':
    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    print(path)
    print("hhhh")
else:
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
dataset = get_dataset(path, args.dataset)    
data = dataset[0]
#data.edge_index, _  = pyg.utils.add_self_loops(data.edge_index)

#import ipdb;ipdb.set_trace()


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
    

#数据处理
features = data.x
features = normalize(features)
features = torch.from_numpy(features)
labels = data.y

adj = torch.eye(data.x.shape[0])
for i in range(data.edge_index.shape[1]):
    adj[data.edge_index[0][i]][data.edge_index[1][i]] = 1
adj = adj.float()
#import ipdb;ipdb.set_trace()
adj = adj_nor(adj)

#import ipdb;ipdb.set_trace()


#train_type

best_model = None
best_val_acc = 0.0
def train(model, optimizer, epoch, features, adj, mask, idx_train, idx_val, labels, label_mask, edge_index, data_aug, encoder_type,train_type, att_type, debias, kk, sample_size, neg_type: int = 0):
    global best_model
    global best_val_acc
    t = time.time()
    model.train()
    optimizer.zero_grad()
    y_pre, _ = model(features, adj, encoder_type)
    loss_train = F.nll_loss(y_pre[idx_train], labels[idx_train])
    acc_train = accuracy(y_pre[idx_train], labels[idx_train])
    
    #sample
    node_mask = torch.empty(features.shape[0],dtype=torch.float32).uniform_(0,1).cuda()
    node_mask = node_mask < sample_size
    
    
    #negative selection
    if neg_type == 0:
        y_pre = y_pre.detach()
        y_pre = y_pre[node_mask]
        #if epoch == 50:
            #import ipdb;ipdb.set_trace()
        
        _, y_poslabel = torch.topk(y_pre, kk)
        y_pl = torch.zeros(y_pre.shape).cuda()
        y_pl = y_pl.scatter_(1, y_poslabel, 1)
        neg_mask = torch.mm(y_pl, y_pl.T) <= 0
        #import ipdb;ipdb.set_trace()
        neg_mask = neg_mask.cuda()
            
        #import ipdb;ipdb.set_trace()
        del y_pl, y_poslabel
        torch.cuda.empty_cache()
        #import ipdb;ipdb.set_trace()
    else :
        pass
    
    train_type = 0
    #import ipdb;ipdb.set_trace()
        
    #true_negmask = torch.min(all_neg, neg_mask)
    #false_negmask = neg_mask - true_negmask
    #true_negmasksum = torch.einsum('ij->',[true_negmask])
    #false_negmasksum = torch.einsum('ij->',[false_negmask])
    #print(true_negmasksum)
    #print(false_negmasksum)
    #if epoch==199:
        #import ipdb;ipdb.set_trace()
    
    if data_aug == 0:
        loss_cl = model.cl_loss(output, mask, node_mask, labels, label_mask, train_type, att_type)
    else:
        features1 = drop_feature(features, 0.3)
        features2 = drop_feature(features, 0.4)
        
        #edge_index1 = dropout_adj(edge_index, p = 0.2)[0]
        #edge_index1, _  = pyg.utils.add_self_loops(edge_index1)
        #adj1 = to_dense_adj(edge_index1)[0]
        #adj1 = adj1.float()
        #adj1 = adj_nor(adj1)
        
        #edge_index2 = dropout_adj(edge_index, p = 0.3)[0]
        #edge_index2, _  = pyg.utils.add_self_loops(edge_index2)
        #adj2 = to_dense_adj(edge_index2)[0]
        #adj2 = adj2.float()
        #adj2 = adj_nor(adj2)
        
        #if adj2.shape[0]<2708:
            #import ipdb;ipdb.set_trace()
        _, output1 = model(features1, adj, encoder_type)
        _, output2 = model(features2, adj, encoder_type)
        
        del features1, features2
        torch.cuda.empty_cache()
        
        loss_cl = model.cl_lossaug(output1, output2, mask, node_mask, labels, neg_mask, train_type, att_type, debias)

    if neg_type == 0 :
        if epoch<=60 :
            loss = loss_train + 0.0001 * loss_cl
        else:
            loss = loss_train + 0.8 * loss_cl
    #loss = loss_train + 0.8 / (1 + math.exp(12 * (50.5 - epoch))) * loss_cl
    else:
        loss = loss_train + args.weight * loss_cl
    #loss = loss_train
    loss.backward()
    optimizer.step()
    
#    with torch.no_grad():
#        model.eval()
#        y_pre, output = model(features, adj, encoder_type)
#        loss_val = F.nll_loss(y_pre[idx_val], labels[idx_val])
#        acc_val = accuracy(y_pre[idx_val], labels[idx_val])
#        if acc_val > best_val_acc:
#            best_val_acc = acc_val
#            model = model.cpu()
#            best_model = dcp(model)
#            model = model.cuda()
            
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        y_pre, _ = model(features, adj, encoder_type)

    loss_val = F.nll_loss(y_pre[idx_val], labels[idx_val])
    acc_val = accuracy(y_pre[idx_val], labels[idx_val])
    if acc_val >= best_val_acc:
        best_val_acc = acc_val
        best_model = dcp(model)
            
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_cl: {:.4f}'.format(loss_cl.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, features, adj, labels, idx_test, encoder_type):
    model.eval()
    y_pre, _ = model(features, adj, encoder_type)
    loss_test = F.nll_loss(y_pre[idx_test], labels[idx_test])
    acc_test = accuracy(y_pre[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test


##SGC    
def propagate(feature, A, order, alpha):
    y = feature
    for i in range(order):
        y = (1 - alpha) * torch.spmm(A, y).detach_() + alpha * y
        
    return y.detach_()
    
if args.encoder == 'SGC':
    features = propagate(features, adj, 2, 0.)
    
#main 
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
data.edge_index = data.edge_index.cuda()
#mask = mask.cuda()
#all_neg = all_neg.cuda()


test_acc = torch.zeros(times)
test_acc = test_acc.cuda()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

for i in range(times):


    index_train = []
    index_val = []
    
    for i_label in range(data.y.max()+1):
        index_sub = [i for i,x in enumerate(data.y) if x==i_label]#train/val index
        index_sub = random.sample(index_sub,60)
        index_train += index_sub[:30]
        index_val += index_sub[30:]
    
    #import ipdb;ipdb.set_trace()
    index_train.sort()
    index_val.sort()
    index_train_val = index_val + index_train
    
    index_test = [i for i in range(data.y.shape[0]) if i not in index_train_val]
    #import ipdb;ipdb.set_trace()
    
    train_mask = sample_mask(index_train, data.y.shape)#array([ True,  True,  True, ..., False, False, False])
    val_mask = sample_mask(index_val, data.y.shape)
    test_mask = sample_mask(index_test, data.y.shape)
    idx_train = torch.Tensor(train_mask).bool()
    idx_val = torch.Tensor(val_mask).bool()
    idx_test = torch.Tensor(test_mask).bool()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


    best_model = None
    best_val_acc = 0.0
    # Model and optimizer
    if args.encoder == 'GCN':
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    tau = args.tau).cuda()
    else:
        model = MLP(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    tau = args.tau).cuda()
                    
    optimizer = optim.Adam(model.parameters(),
                lr=args.lr, weight_decay=args.weight_decay)
    
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(model, optimizer, epoch, features, adj, None, idx_train, idx_val, labels, None, data.edge_index,args.data_aug, args.encoder_type,args.train_type, args.att_type, args.debias, args.kk, args.sample_size)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing
    test_acc[i] = test(best_model, features, adj, labels, idx_test, args.encoder_type)


print("=== Final ===")
print(torch.max(test_acc))
print(torch.min(test_acc))
#print("30次平均",torch.mean(test_acc))
#print("30次标准差",test_acc.std())
#print("20次平均",torch.mean(test_acc[:20]))
#print("20次标准差",test_acc[:20].std())
print("10次平均",torch.mean(test_acc))
print("10次标准差",test_acc.std())
#import ipdb;ipdb.set_trace()

print(test_acc)
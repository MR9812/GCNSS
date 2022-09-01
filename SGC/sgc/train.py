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
from models import GCN, MLP, GAT

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
parser.add_argument('--neg_type', type=float, default=0,
                    help='0,selection;1 not selection')
parser.add_argument('--encoder_type', type=int, default=3,
                    help='do data augmentation.')
parser.add_argument('--debias', type=float, default=0.,
                    help='debias rate.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
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
def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP','WikiCS','Amazon-Photo']
    name = 'dblp' if name == 'DBLP' else name
    print(name)
    return (CitationFull if name == 'dblp' else Planetoid)(path,name,transform=T.NormalizeFeatures())

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



#数据处理
idx_train = data.train_mask
idx_val = data.val_mask
idx_test = data.test_mask
features = data.x
features = normalize(features)
features = torch.from_numpy(features)
labels = data.y
adj = torch.eye(data.x.shape[0])
for i in range(data.edge_index.shape[1]):
    adj[data.edge_index[0][i]][data.edge_index[1][i]] = 1
adj = adj.float()
adj = adj_nor(adj)



#全对  t_mask全对的正例矩阵，all_neg全对的负例矩阵，neg_mask不做选择的负例矩阵
labels_t = data.y.contiguous().view(-1,1)
t_mask = torch.eq(labels_t, labels_t.T).float()
all_neg = torch.where(t_mask>0, 0, 1)
all_negsum = torch.einsum('ij->',[all_neg])
neg_mask = torch.ones(features.shape[0],features.shape[0])-torch.eye(features.shape[0])
true_neg = torch.min(all_neg, neg_mask)
false_neg = neg_mask - true_neg
true_negsum = torch.einsum('ij->',[true_neg])
false_negsum = torch.einsum('ij->',[false_neg])
print(true_negsum)
print(false_negsum)
print(all_negsum)


#label_mask
#140 0.9 无用
labels_train = labels[idx_train]
label_mask = torch.randn(len(labels_train), torch.max(data.y)+1)
label_mask = label_mask.uniform_(0,1)
for j in range(len(labels_train)):
    label_mask[j][labels_train[j]]=0
label_masksum = torch.sum(label_mask, dim=1)
label_masksum = label_masksum.view(-1,1)
label_mask = label_mask / (label_masksum * 10)
for j in range(len(labels_train)):
    label_mask[j][labels_train[j]]=0.9


#mask正例
if args.cl_num == 0:
    pass
elif args.cl_num ==1:
    pass
elif args.cl_num ==2:
    mask = torch.eye(data.x.shape[0])
    args.train_type = 0
else:
    pass

best_model = None
best_val_acc = 0.0
def train(model, optimizer, epoch, features, adj, mask, idx_train, idx_val, labels, label_mask, neg_mask, all_neg, edge_index, data_aug, encoder_type,train_type, att_type, debias, kk, sample_size, neg_type):
    global best_model
    global best_val_acc
    t = time.time()
    model.train()
    optimizer.zero_grad()
    y_pre, _ = model(features, adj, encoder_type)
    #import ipdb;ipdb.set_trace()
    loss_train = F.nll_loss(y_pre[idx_train], labels[idx_train])
    acc_train = accuracy(y_pre[idx_train], labels[idx_train])
    
    #sample
    node_mask = torch.empty(features.shape[0],dtype=torch.float32).uniform_(0,1).cuda()
    node_mask = node_mask < sample_size
    
    #negative selection
    if neg_type == 0:
        y_pre = y_pre.detach()
        #y_pre = y_pre[node_mask]
        #if epoch == 50:
            #import ipdb;ipdb.set_trace()
        
        _, y_poslabel = torch.topk(y_pre, kk)
        y_pl = torch.zeros(y_pre.shape).cuda()
        y_pl = y_pl.scatter_(1, y_poslabel, 1)
        neg_mask = torch.where(torch.mm(y_pl, y_pl.T) <= 0,1,0)
        
        neg_mask = neg_mask[node_mask].T
        neg_mask = neg_mask[node_mask].T
        
        neg_mask = neg_mask.cuda()
        del y_pl, y_poslabel
        torch.cuda.empty_cache()
        #import ipdb;ipdb.set_trace()
    else :
        pass
    
    
    train_type = 0
    #if epoch==50:
        #import ipdb;ipdb.set_trace()
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
        #features1 = aug_random_mask(features, 0.3)
        #features2 = aug_random_mask(features, 0.4)
        features1 = drop_feature(features, 0.3)
        features2 = drop_feature(features, 0.4)
        #import ipdb;ipdb.set_trace()
        
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
        
        if neg_type == 0:
            loss_cl = model.cl_lossaug(output1, output2, mask, node_mask, labels, neg_mask, train_type, att_type, debias)
        else:
            loss_cl = model.cl_lossaug(output1, output2, mask, node_mask, labels, neg_mask, train_type, att_type, debias, 0)
    if neg_type == 0:    
        if epoch<=50:
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
#    acc_test = accuracy(y_pre[idx_test], labels[idx_test])
    if acc_val > best_val_acc:
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
def propagate1(feature, A, order, alpha):
    y = feature
    out = feature
    for i in range(order):
        y = torch.spmm(A, y).detach_()
        out = out + y
        
    return out.detach_()/(order + 1)
    
def propagate(feature, A, order, alpha):
    y = feature
    for i in range(order):
        y = (1 - alpha) * torch.spmm(A, y).detach_() + alpha * y
        
    return y.detach_()
    
def propagate2(features, adj, degree, alpha):
    ori_features = features
    emb = alpha * features
    for i in range(degree):
        features = torch.spmm(adj, features)
        emb = emb + (1-alpha)*features/degree
    return emb
    
if args.encoder == 'SGC':
    features = propagate(features, adj, 2, 0.)
#    features = features
#    features = torch.from_numpy(features)
#    pass
#if args.encoder == 'SGC':
#    features = data.x
#    adj = data.edge_index

#main 
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
data.edge_index = data.edge_index.cuda()
neg_mask = neg_mask.cuda()
if args.dataset == "PubMed":
    pass
else:
    mask = mask.cuda()
    label_mask = label_mask.cuda()
    all_neg = all_neg.cuda()


test_acc = torch.zeros(times)
test_acc = test_acc.cuda()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

for i in range(times):
    best_model = None
    best_val_acc = 0.0
    # Model and optimizer
    if args.encoder == 'GCN':
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    tau = args.tau).cuda()
    elif args.encoder == 'GAT':
        model = GAT(num_features=features.shape[1], 
                    hidden_channels=16, 
                    num_classes=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    nheads=8, 
                    alpha=0.2,
                    tau = 0.4)
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
        if args.dataset == "PubMed":
            train(model, optimizer, epoch, features, adj, None, idx_train, idx_val, labels, None, neg_mask, None, data.edge_index,args.data_aug
        ,args.encoder_type,args.train_type, args.att_type, args.debias, args.kk, args.sample_size, args.neg_type)
        else:
            train(model, optimizer, epoch, features, adj, mask, idx_train, idx_val, labels, label_mask, neg_mask, all_neg, data.edge_index,args.data_aug
        ,args.encoder_type,args.train_type, args.att_type, args.debias, args.kk, args.sample_size, args.neg_type)
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
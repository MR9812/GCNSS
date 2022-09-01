from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models_dgl import GAT
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull,WikiCS, Coauthor, Amazon
import os.path as osp
import scipy.sparse as sp

import argparse
import networkx as nx
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from dgl import DGLGraph

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='False pygGAT; True spGAT')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--out_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dataset', type=str, default='Cora', help='Cora/CiteSeer/PubMed/')
parser.add_argument("--attn-drop", type=float, default=.6, help="attention dropout")
parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU.")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    
def adj_nor(edge):
    degree = torch.sum(edge, dim=1)
    degree = 1 / torch.sqrt(degree)
    degree = torch.diag(degree)
    adj = torch.mm(torch.mm(degree, edge), degree)
    return adj



#load dataset
if args.dataset == 'cora':
    data = CoraGraphDataset()
elif args.dataset == 'citeseer':
    data = CiteseerGraphDataset()
elif args.dataset == 'pubmed':
    data = PubmedGraphDataset()
elif args.dataset == 'coauthor-cs':
    data = CoauthorCSDataset()
elif args.dataset == 'coauthor-phy':
    data = CoauthorPhysicsDataset()
else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))

g = data[0]
features = g.ndata['feat']
labels = g.ndata['label']
#import ipdb;ipdb.set_trace()
if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
    idx_train = g.ndata['train_mask']
    idx_val = g.ndata['val_mask']
    idx_test = g.ndata['test_mask']
    
num_feats = features.shape[1]
n_classes = labels.max() + 1
#n_edges = data.graph.number_of_edges()
g = dgl.remove_self_loop(g)
#g = DGLGraph(g)
g = dgl.add_self_loop(g)
#n_edges = g.number_of_edges()
heads = ([args.nb_heads] * 1) + [args.out_heads]
#13264

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)





def train(epoch, model, optimizer, features, idx_train, idx_val, idx_test):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features)    
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.8f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.8f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'acc_test: {:.4f}'.format(acc_test.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item(), acc_test.data.item()


def compute_test(model, features, adj, idx_test):
    model.eval()
    output = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
          
    return acc_test.item()




times = 10
test_acc = torch.zeros(times)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


for i in range(times):
    if args.dataset == 'coauthor-cs' or args.dataset == 'coauthor-phy':
        index_train = []
        index_val = []
        
        for i_label in range(labels.max()+1):
            index_sub = [i for i,x in enumerate(labels) if x==i_label]#train/val index
            index_sub = random.sample(index_sub,60)
            index_train += index_sub[:30]
            index_val += index_sub[30:]
        
        #import ipdb;ipdb.set_trace()
        index_train.sort()
        index_val.sort()
        index_train_val = index_val + index_train
        
        index_test = [i for i in range(labels.shape[0]) if i not in index_train_val]
        #import ipdb;ipdb.set_trace()
        
        train_mask = sample_mask(index_train, labels.shape)#array([ True,  True,  True, ..., False, False, False])
        val_mask = sample_mask(index_val, labels.shape)
        test_mask = sample_mask(index_test, labels.shape)
        idx_train = torch.Tensor(train_mask).bool()
        idx_val = torch.Tensor(val_mask).bool()
        idx_test = torch.Tensor(test_mask).bool()
#        idx_train = idx_train.cuda()
#        idx_val = idx_val.cuda()
#        idx_test = idx_test.cuda()
    # Model and optimizer
    if args.gpu < 0:
        args.cuda = False
    else:
        args.cuda = True
        g = g.int().to(args.gpu)
        #torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
#        g = g.cuda()
        
    if args.sparse:
        model = SpGAT(nfeat=features.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha)
    else:
        model = GAT(g,
                    1,
                    num_feats, 
                    args.hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.dropout, 
                    args.attn_drop,
                    args.alpha,
                    args.residual)
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.weight_decay)
    
    if args.cuda:
        model.cuda()

    
    
    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    best_acc = 0
    for epoch in range(args.epochs):
        loss_e, acc_e = train(epoch, model, optimizer, features, idx_train, idx_val, idx_test)
        loss_values.append(loss_e)
    
#        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
            best_acc = acc_e
        else:
            bad_counter += 1
    
        if bad_counter == args.patience:
            break
    
#        files = glob.glob('*.pkl')
#        for file in files:
#            epoch_nb = int(file.split('.')[0])
#            if epoch_nb < best_epoch:
#                os.remove(file)
    
#    files = glob.glob('*.pkl')
#    for file in files:
#        epoch_nb = int(file.split('.')[0])
#        if epoch_nb > best_epoch:
#            os.remove(file)
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    # Restore best model
#    print('Loading {}th epoch'.format(best_epoch))
#    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    
    # Testing
    #import ipdb;ipdb.set_trace()
    print(best_acc)
    test_acc[i] = best_acc


print("=== Final ===")
print(torch.max(test_acc))
print(torch.min(test_acc))
print("10次平均",torch.mean(test_acc))
print("10次标准差",test_acc.std())

print(test_acc)
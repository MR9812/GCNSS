import arguments
import numpy as np

import torch
import torch.nn.functional as F

from utils import accuracy, load_data
from models import APPNP1
import random
from early_stop import EarlyStopping, Stop_args

import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull,WikiCS, Coauthor, Amazon
import torch_geometric as pyg
import torch_geometric.transforms as T
import scipy.sparse as sp

#adj normalization
def adj_nor(edge):
    degree = torch.sum(edge, dim=1)
    degree = 1 / torch.sqrt(degree)
    degree = torch.diag(degree)
    adj = torch.mm(torch.mm(degree, edge), degree)
    return adj
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


args = arguments.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#load dataset
def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP','WikiCS','Amazon-Photo']
    name = 'dblp' if name == 'DBLP' else name
    print(name)
    return (CitationFull if name == 'dblp' else Planetoid)(path,name,transform=T.NormalizeFeatures())
    
    
#load dataset
def get_dataset2(path, name):
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
    dataset = get_dataset(path, args.dataset) 
else:
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset2(path, args.dataset) 

data = dataset[0]
#data.edge_index, _  = pyg.utils.add_self_loops(data.edge_index)
#import ipdb;ipdb.set_trace()


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



#数据处理
if args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed':
    idx_train = g.ndata['train_mask']
    idx_val = g.ndata['val_mask']
    idx_test = g.ndata['test_mask']

features = data.x
features = normalize(features)
features = torch.from_numpy(features)
labels = data.y
adj = torch.eye(data.x.shape[0])
for i in range(data.edge_index.shape[1]):
    adj[data.edge_index[0][i]][data.edge_index[1][i]] = 1
adj = adj.float()
adj = adj_nor(adj)




# Load data and pre_process data 
#adj, features, labels, idx_train, idx_val, idx_test = load_data(graph_name = args.dataset, str_noise_rate=args.str_noise_rate, seed = args.seed)


def train(epoch, sample_size, debias, kk: int = 1):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + args.weight_decay * torch.sum(model.Linear1.weight ** 2) / 2
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss = loss_train
    loss.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'loss_val: {:.4f}'.format(loss_val.item()),
        'acc_val: {:.4f}'.format(acc_val.item()))

    return loss_val.item(), acc_val.item()

def test():
    model.eval()
    output = model(features, adj)
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
torch.manual_seed(args.ini_seed)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(args.ini_seed)


for i in range(times):
    if args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed':
        pass
    else:
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
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
    
    
    # Model and optimizer
    model = APPNP1(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout, 
                K=args.K, 
                alpha=args.alpha).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),
                           lr=args.lr)
    
    features = features.to(device)
#    adj = adj.to_sparse().requires_grad_(True)
    adj = adj.to(device)
    labels = labels.to(device)
    
    
    
    
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)
    for epoch in range(args.epochs):
        loss_val, acc_val = train(epoch, args.sample_size, args.debias)
        if early_stopping.check([acc_val, loss_val], epoch):
            break
    
    print("Optimization Finished!")
    
    # Restore best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    model.load_state_dict(early_stopping.best_state)
    test_acc[i] = test()

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

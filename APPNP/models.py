import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sp
import math
import random

class Linear(nn.Module): 
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MLP(nn.Module):#
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return torch.log_softmax(self.Linear2(x), dim=1)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=True)

    def forward(self, x, adj):
        x = torch.relu(self.Linear1(torch.matmul(adj, x)))
        h = self.Linear2(torch.matmul(adj, x))
        return torch.log_softmax(h, dim=-1)


class SGCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(SGCN, self).__init__()
        self.Linear = Linear(nfeat, nclass, dropout, bias=False)
        self.x = None

    def forward(self, x, adj): 
        if self.x == None: 
            self.x = torch.matmul(adj, torch.matmul(adj, x))
        return torch.log_softmax(self.Linear(self.x), dim=-1)

class APPNP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, K, alpha, tau):
        super(APPNP, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=False)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=False)
        self.alpha = alpha
        self.K = K
        self.fc1 = torch.nn.Linear(nhid, nhid)
        self.fc2 = torch.nn.Linear(nhid, nhid)
        self.tau = tau

    def forward(self, x, adj):
#        x = torch.relu(self.Linear1(x))
#        h0 = self.Linear2(x)
#        h = h0
#        for _ in range(self.K):
 #           h = (1 - self.alpha) * torch.matmul(adj, h) + self.alpha * h0
        
#        x0 = x
#        x_cl = x0
#        for _ in range(self.K):
#            x_cl = (1 - self.alpha) * torch.matmul(adj, x_cl) + self.alpha * x0 

        h0 = torch.relu(self.Linear1(x))
        x_cl = h0
        for _ in range(self.K):
            x_cl = (1 - self.alpha) * torch.matmul(adj, x_cl) + self.alpha * h0
        h = self.Linear2(x_cl)
        
        
        return torch.log_softmax(h, dim=-1), x_cl
        
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        
    def suplabel_lossv6neg(self, z1: torch.Tensor, z2: torch.Tensor , mask: torch.Tensor, neg_mask: torch.Tensor, debias, mean_type: int = 1):
        if mean_type == 0:
            s_value = torch.mm(z1 , z1.t())
            b_value = torch.mm(z1 , z2.t())
            s_value_max, _ = torch.max(s_value, dim=1, keepdim=True)
            s_value = s_value - s_value_max.detach()
            b_value_max, _ = torch.max(b_value, dim=1, keepdim=True)
            b_value = b_value - b_value_max.detach()
            s_value = torch.exp(s_value / self.tau)
            b_value = torch.exp(b_value / self.tau)
        else:
            s_value = torch.exp(torch.mm(z1 , z1.t()) / self.tau)
            b_value = torch.exp(torch.mm(z1 , z2.t()) / self.tau)
        #import ipdb;ipdb.set_trace()
        value_zi = b_value.diag().unsqueeze(0).T
        
        #import ipdb;ipdb.set_trace()
        value_neg = (s_value + b_value) * neg_mask.float()
        value_neg = value_neg.sum(dim=1, keepdim=True)
        neg_sum = 2 * neg_mask.sum(dim=1, keepdim=True)
        value_neg = (value_neg - value_zi * neg_sum * debias) / (1 - debias)
        value_neg = torch.max(value_neg, neg_sum * math.exp(-1.0 / self.tau))
        value_mu = value_zi + value_neg
        
        #import ipdb;ipdb.set_trace()
        loss = -torch.log(value_zi / value_mu)
        return loss
        
    
    def cl_lossaug(self, z1: torch.Tensor, z2: torch.Tensor, mask: torch.Tensor, train_mask: torch.Tensor, labels, neg_mask, 
            train_type, att_type, debias, neg: int = 1, mean: bool = True ):
#neg   train8.py=1  train11.py=0
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        #import ipdb;ipdb.set_trace()
        if train_type==0:
            labels = labels[train_mask]
            h1 = h1[train_mask]
            h2 = h2[train_mask]
            if neg==0:
                neg_mask = neg_mask[train_mask].T
                neg_mask = neg_mask[train_mask].T
            #neg_sample = torch.empty(neg_mask.shape,dtype=torch.float32).uniform_(0,1).cuda()
            #neg_sample = torch.where(neg_sample<0.857,1,0)
            #neg_mask = neg_mask * neg_sample
            #import ipdb;ipdb.set_trace()
        if att_type == 0:
            pass
        else:
            loss1 = self.suplabel_lossv6neg(h1, h2, mask, neg_mask, debias)
            loss2 = self.suplabel_lossv6neg(h2, h1, mask, neg_mask, debias)
            ret = (loss1 + loss2) / 2

        ret = ret.mean() if mean else ret.sum()
        return ret

class PT(nn.Module): 
    def __init__(self, nfeat, nhid, nclass, dropout, epsilon, mode, K, alpha):
        # mode: 0-PTS, 1-PTS, 2-PTA
        super(PT, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=True)
        self.epsilon = epsilon
        self.mode = mode
        self.K = K 
        self.alpha = alpha
        self.number_class = nclass

    def forward(self, x): 
        x = torch.relu(self.Linear1(x))
        return self.Linear2(x)

    def loss_function(self, y_hat, y_soft, epoch = 0): 
        if self.training: 
            y_hat_con = torch.detach(torch.softmax(y_hat, dim=-1))
            exp = np.log(epoch / self.epsilon + 1)
            if self.mode == 2: 
                loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), torch.mul(y_soft, y_hat_con**exp))) / self.number_class  # PTA
            elif self.mode == 1:
                loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), torch.mul(y_soft, y_hat_con))) / self.number_class  # PTD
            else: 
                loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft)) / self.number_class # PTS
        else: 
            loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft)) / self.number_class
        return loss

    def inference(self, h, adj): 
        y0 = torch.softmax(h, dim=-1) 
        y = y0
        for i in range(self.K):
            y = (1 - self.alpha) * torch.matmul(adj, y) + self.alpha * y0
        return y
        
        
class APPNP1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, K, alpha):
        super(APPNP1, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=False)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=False)
        self.alpha = alpha
        self.K = K

    def forward(self, x, adj):
        x = torch.relu(self.Linear1(x))
        h0 = self.Linear2(x)
        h = h0
        for _ in range(self.K):
            h = (1 - self.alpha) * torch.matmul(adj, h) + self.alpha * h0
        return torch.log_softmax(h, dim=-1)

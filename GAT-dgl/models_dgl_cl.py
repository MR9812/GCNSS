import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GATConv
import math


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop1,
                 feat_drop2,
                 attn_drop,
                 negative_slope,
                 residual,
                 tau = 0.4):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop1, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop2, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop2, attn_drop, negative_slope, residual, None))
        
        self.fc1 = torch.nn.Linear(num_hidden*heads[0], num_hidden*heads[0])
        self.fc2 = torch.nn.Linear(num_hidden*heads[0], num_hidden*heads[0])
        self.tau = tau

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        
        output = h
        
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return F.log_softmax(logits, dim=1), h
        
        
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
            h1 = h1[train_mask]
            h2 = h2[train_mask]
        if att_type == 0:
            pass
        else:
            loss1 = self.suplabel_lossv6neg(h1, h2, mask, neg_mask, debias)
            loss2 = self.suplabel_lossv6neg(h2, h1, mask, neg_mask, debias)
            ret = (loss1 + loss2) / 2

        ret = ret.mean() if mean else ret.sum()
        return ret   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
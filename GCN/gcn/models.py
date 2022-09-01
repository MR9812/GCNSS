import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import math
from torch_geometric.nn import GATConv, GCNConv, SGConv
from torch_geometric.nn.inits import glorot, zeros


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, tau):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.fc3 = nn.Linear(nclass, 128)
        self.fc4 = torch.nn.Linear(nhid, nhid)
        self.fc5 = torch.nn.Linear(nhid, nhid)
        self.fc6 = torch.nn.Linear(nhid, nclass)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
        self.tau = tau
        self.nclass = nclass

    def forward(self, x, adj, encoder_type):
        if encoder_type == 0:
            #GCN + W(7*128)
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)

            out = F.relu(x)
            out = F.dropout(out, self.dropout, training=self.training)
            out = self.fc3(out)
        elif encoder_type == 1:
            #2GCN + W(128*7)
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            out = self.gc3(x, adj)
            x = F.relu(out)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.fc6(x)
        elif encoder_type == 2:
            #gcn1 cl  gcn2
            out = self.gc1(x, adj)
            x = F.relu(out)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
        else :
            #AAXW0 cl  AAXWW1 CE
            #out = self.gc1(x, adj)
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            out = torch.spmm(adj , x)
            x = self.fc6(out)

        return F.log_softmax(x, dim=1), out


    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc4(z))
        return self.fc5(z)


    def suplabel_lossv6neg(self, z1: torch.Tensor, z2: torch.Tensor , neg_mask: torch.Tensor, debias):
        
        s_value = torch.exp(torch.mm(z1 , z1.t()) / self.tau)
        b_value = torch.exp(torch.mm(z1 , z2.t()) / self.tau)
        
        value_zi = b_value.diag().unsqueeze(0).T
        value_neg = (s_value + b_value) * neg_mask.float()
        value_neg = value_neg.sum(dim=1, keepdim=True)
        neg_sum = 2 * neg_mask.sum(dim=1, keepdim=True)
        value_neg = (value_neg - value_zi * neg_sum * debias) / (1 - debias)
        value_neg = torch.max(value_neg, neg_sum * math.exp(-1.0 / self.tau))
        value_mu = value_zi + value_neg
        
        loss = -torch.log(value_zi / value_mu)
        return loss


    def cl_lossaug(self, z1: torch.Tensor, z2: torch.Tensor, train_mask: torch.Tensor, neg_mask, debias, mean: bool = True ):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        h1 = h1[train_mask]
        h2 = h2[train_mask]
        
        loss1 = self.suplabel_lossv6neg(h1, h2, neg_mask, debias)
        loss2 = self.suplabel_lossv6neg(h2, h1, neg_mask, debias)
        ret = (loss1 + loss2) / 2

        ret = ret.mean() if mean else ret.sum()
        return ret






class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, tau):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Linear(nfeat, nhid)
        self.fc2 = torch.nn.Linear(nhid, nclass)
        self.fc3 = torch.nn.Linear(nhid, nhid)
        self.fc4 = torch.nn.Linear(nhid, nhid)
        self.dropout = dropout
        self.tau = tau
        self.nclass = nclass

    def forward(self, x, adj, encoder_type):
        out = self.fc1(x)
        x = F.relu(out)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1), out


    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc3(z))
        return self.fc4(z)


    def suplabel_lossv6neg(self, z1: torch.Tensor, z2: torch.Tensor , neg_mask: torch.Tensor, debias):
        
        s_value = torch.exp(torch.mm(z1 , z1.t()) / self.tau)
        b_value = torch.exp(torch.mm(z1 , z2.t()) / self.tau)
        
        value_zi = b_value.diag().unsqueeze(0).T
        value_neg = (s_value + b_value) * neg_mask.float()
        value_neg = value_neg.sum(dim=1, keepdim=True)
        neg_sum = 2 * neg_mask.sum(dim=1, keepdim=True)
        value_neg = (value_neg - value_zi * neg_sum * debias) / (1 - debias)
        value_neg = torch.max(value_neg, neg_sum * math.exp(-1.0 / self.tau))
        value_mu = value_zi + value_neg
        
        loss = -torch.log(value_zi / value_mu)
        return loss


    def cl_lossaug(self, z1: torch.Tensor, z2: torch.Tensor, train_mask: torch.Tensor, neg_mask, debias, mean: bool = True ):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        h1 = h1[train_mask]
        h2 = h2[train_mask]
        
        loss1 = self.suplabel_lossv6neg(h1, h2, neg_mask, debias)
        loss2 = self.suplabel_lossv6neg(h2, h1, neg_mask, debias)
        ret = (loss1 + loss2) / 2

        ret = ret.mean() if mean else ret.sum()
        return ret








                                  
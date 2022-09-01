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

    def sup_loss(self, z: torch.Tensor, mask: torch.Tensor, mean_type: int = 0):
        e = torch.eye(z.shape[0])
        one = torch.ones(z.shape[0],1)
        e = e.cuda()
        one = one.cuda()
        value = torch.mm(z , z.t())
        value = torch.exp(value / self.tau)
        #import ipdb;ipdb.set_trace()
        p = torch.mm(mask, one)
        p = p - 1
        p = 1/p

        value_mu = torch.mm(value, one) - torch.mm(value * e, one)
        value_zi = torch.mm(value * (mask - e), one)

        if mean_type == 0:
            loss = -torch.log(p * value_zi / value_mu)
        else:
            loss = -p * torch.log(value_zi/value_mu)
        #import ipdb;ipdb.set_trace()
        return loss

    def suplabel_loss(self, z: torch.Tensor, mask: torch.Tensor, labels, mean_type: int = 0):
        label_mask = torch.zeros(z.shape[0], self.nclass)
        label_mask = label_mask.cuda()
        for j in range(z.shape[0]):
            label_mask[j][labels[j]]=1

        e = torch.eye(z.shape[0])
        one = torch.ones(z.shape[0],1)
        e = e.cuda()
        one = one.cuda()

        value = torch.mm(z , label_mask.t())
        value = torch.exp(value / self.tau)
        p = len(labels)/self.nclass

        value_zi = torch.mm(value * e, one)
        value_mu = torch.mm(value, one) / p

        if mean_type == 0:
            loss = -torch.log(value_zi / value_mu)
        else:
            loss = -torch.log(value_zi / value_mu)
        #import ipdb;ipdb.set_trace()
        return loss

    def suplabel_lossv2(self, z: torch.Tensor, mask: torch.Tensor, labels, mean_type: int = 0):
        label_mask = torch.zeros(z.shape[0], self.nclass)
        label_mask = label_mask.cuda()
        for j in range(z.shape[0]):
            label_mask[j][labels[j]]=1

        e = torch.eye(z.shape[0])
        one = torch.ones(z.shape[0],1)
        e = e.cuda()
        one = one.cuda()

        value1 = torch.mm(z , z.t())
        value1 = torch.exp(value1 / self.tau)
        value2 = torch.mm(z , label_mask.t())
        value2 = torch.exp(value2 / self.tau)
        #import ipdb;ipdb.set_trace()
        p = torch.mm(mask, one)
        p = 2 * p - 1
        p = 1/p
        q = len(labels)/self.nclass

        value_mu = torch.mm(value1, one) - torch.mm(value1 * e, one) + torch.mm(value2 * e, one) * q
        value_zi = torch.mm(value1 * (mask - e), one) + torch.mm(value2, one)

        if mean_type == 0:
            loss = -torch.log(p * value_zi / value_mu)
        else:
            loss = -p * torch.log(value_zi/value_mu)

        return loss

    def suplabel_lossv3(self, z: torch.Tensor, mask: torch.Tensor, label_mask, mean_type: int = 0):
        e = torch.eye(z.shape[0])
        one = torch.ones(z.shape[0],1)
        e = e.cuda()
        one = one.cuda()
        #import ipdb;ipdb.set_trace()
        value = torch.mm(z , label_mask.t())
        value = torch.exp(value / self.tau)
        #import ipdb;ipdb.set_trace()
        p = torch.mm(mask, one)
        p = 1/p

        value_mu = torch.mm(value, one)
        value_zi = torch.mm(value * mask, one)

        if mean_type == 0:
            loss = -torch.log(p * value_zi / value_mu)
        else:
            loss = -p * torch.log(value_zi/value_mu)
        #import ipdb;ipdb.set_trace()
        return loss

    def suplabel_lossv4(self, z: torch.Tensor, mask: torch.Tensor, label_mask, mean_type: int = 0):
        e = torch.eye(z.shape[0])
        one = torch.ones(z.shape[0],1)
        e = e.cuda()
        one = one.cuda()
        value1 = torch.mm(z , z.t())
        value1 = torch.exp(value1 / self.tau)
        value2 = torch.mm(z , label_mask.t())
        value2 = torch.exp(value2 / self.tau)
        #import ipdb;ipdb.set_trace()
        p = torch.mm(mask, one)
        p = 1 / (2*p-1)

        value_mu = torch.mm(value2, one) + torch.mm(value1, one) - torch.mm(value1 * e, one)
        value_zi = torch.mm(value2 * mask, one) + torch.mm(value1 * (mask - e), one)

        if mean_type == 0:
            loss = -torch.log(p * value_zi / value_mu)
        else:
            loss = -p * torch.log(value_zi/value_mu)
        #import ipdb;ipdb.set_trace()
        return loss

    def suplabel_lossv5(self, z: torch.Tensor, mask: torch.Tensor, mean_type: int = 0):
        e = torch.eye(z.shape[0])
        one = torch.ones(z.shape[0],1)
        e = e.cuda()
        one = one.cuda()
        zero = torch.zeros(z.shape[0],z.shape[0])
        zero = zero.cuda()
        mask = torch.mm(z , z.t())

        mask = torch.where(mask>0,mask,zero)
        value = torch.exp(mask / self.tau)

        mask1 = mask * (one - e)
        masksum = torch.sum(mask1, dim=1)
        masksum = masksum.view(-1,1)
        mask2 = mask1 / masksum

        value_mu = torch.mm(value, one) - torch.mm(value * e, one)
        value_zi = torch.mm(value * mask2, one)

        loss = -torch.log(value_zi / value_mu)

        return loss

    def suplabel_lossv6(self, z1: torch.Tensor, z2: torch.Tensor , mask: torch.Tensor, mean_type: int = 0):
        e = torch.eye(z1.shape[0])
        one = torch.ones(z1.shape[0],1)
        e = e.cuda()
        one = one.cuda()
        s_value = torch.mm(z1 , z1.t())
        b_value = torch.mm(z1 , z2.t())
        s_value = torch.exp(s_value / self.tau)
        b_value = torch.exp(b_value / self.tau)
        #import ipdb;ipdb.set_trace()
        p = torch.mm(mask, one)
        p = 2 * p - 1
        p = 1/p

        value_mu = torch.mm(s_value, one) + torch.mm(b_value, one) - torch.mm(s_value * e, one)
        value_zi = torch.mm(s_value * (mask - e), one) + torch.mm(b_value * mask, one)

        if mean_type == 0:
            loss = -torch.log(p * value_zi / value_mu)
        else:
            loss = -p * torch.log(value_zi/value_mu)

        return loss


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


    def cl_loss(self, z: torch.Tensor, mask: torch.Tensor, train_mask: torch.Tensor, labels, label_mask,
             train_type: int = 0, att_type: int = 0,mean: bool = True):
        #h = self.projection(z)
        h = F.normalize(z)
        if train_type==0:
            #import ipdb;ipdb.set_trace()
            labels = labels[train_mask]
            h = h[train_mask]
        if att_type == 0:
            ret = self.sup_loss(h , mask)
            #import ipdb;ipdb.set_trace()
        elif att_type == 1:
            ret = self.suplabel_loss(h , mask, labels)
        elif att_type == 2:
            ret = self.suplabel_lossv2(h , mask, labels)
            #l1 = self.sup_loss_neg(h1, h2 , mask, neg_mask)
            #l2 = self.sup_loss_neg(h2, h1 , mask, neg_mask)
            #ret = (l1 + l2) * 0.5
        elif att_type == 3:
            ret = self.suplabel_lossv3(h , mask, label_mask)
        elif att_type ==4:
            ret = self.suplabel_lossv4(h , mask, label_mask)
        else:
            ret = self.suplabel_lossv5(h , mask, label_mask)

        ret = ret.mean() if mean else ret.sum()
        return ret

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
            loss1 = self.suplabel_lossv6(h1, h2, mask)
            loss2 = self.suplabel_lossv6(h2, h1, mask)
            ret = (loss1 + loss2) / 2
        else:
            loss1 = self.suplabel_lossv6neg(h1, h2, mask, neg_mask, debias)
            loss2 = self.suplabel_lossv6neg(h2, h1, mask, neg_mask, debias)
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
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        if train_type==0:
            labels = labels[train_mask]
            h1 = h1[train_mask]
            h2 = h2[train_mask]
        if att_type == 0:
            loss1 = self.suplabel_lossv6(h1, h2, mask)
            loss2 = self.suplabel_lossv6(h2, h1, mask)
            ret = (loss1 + loss2) / 2
        else:
            loss1 = self.suplabel_lossv6neg(h1, h2, mask, neg_mask, debias)
            loss2 = self.suplabel_lossv6neg(h2, h1, mask, neg_mask, debias)
            ret = (loss1 + loss2) / 2

        ret = ret.mean() if mean else ret.sum()
        return ret
        
        
        
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout, alpha, nheads, tau):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(
            8 * hidden_channels, num_classes, heads=1, concat=True, dropout=0.6
        )
        self.fc1 = torch.nn.Linear(hidden_channels*nheads, hidden_channels*nheads)
        self.fc2 = torch.nn.Linear(hidden_channels*nheads, hidden_channels*nheads)
        self.tau = tau

    def forward(self, x, edge_index, encoder_type):
        x = F.dropout(x, p=0.6, training=self.training)
        out = self.conv1(x, edge_index)
        x = F.elu(out)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), out
    
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
#sup_loss,向量标签分布的一致性
#suplabel_loss，真实label的one-hot,1正例,6负例
#suplabel_lossv2，1+6-加上标签分布的约束
#suplabel_lossv3，label smoothing,多正多负
#suplabel_lossv4，label smoothing多正多负  加上   标签平滑
#标签传播
#多negative
#self.suplabel_lossv6  数据增强的两组向量互相对比学习
#self.suplabel_lossv6neg 数据增强的对比学习 ，去除假正例
#CUDA_VISIBLE_DEVICES=1 python train3.py --data_aug 1 --hidden 128 --encoder_type 3 --cl_num 2  --att_type 1  --weight 0.4
#CUDA_VISIBLE_DEVICES=2 python train5.py --encoder_type 3 --data_aug 1 --cl_num 2 --att_type 1 --hidden 128 --weight 0.0001 --dataset CiteSeer
                                            
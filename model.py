import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.nn import Parameter
from torch_scatter import scatter_add
import math
import numpy as np
from torch.autograd import Variable


class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_channels)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # 偏置项bias的值初始化在区间 [-stdv, stdv] 内。
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        adj = self.nomarlizeAdj(adj)
        m = torch.mm(x, self.weight)
        m = torch.spmm(adj, m)
        return m

    def normalizeLelf(self, adj):
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj
        return adj

        # D-1/2 * A * D-1/2

    def nomarlizeAdj(self, adj):
        adj = adj.to('cuda:0')
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = row_sum.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        row_indices = indices[0, :]
        col_indices = indices[1, :]
        new_values = deg_inv_sqrt[row_indices] * values * deg_inv_sqrt[col_indices]
        normalized_adj = torch.sparse_coo_tensor(indices, new_values, adj.size())
        return normalized_adj


class EC(torch.nn.Module):

    def __init__(self, in_channels, out_channels, alpha, gamma, dropout=0.1):
        super(EC, self).__init__()
        self.bias = False
        self.alpha = alpha
        self.gamma = gamma
        self.d_liner = nn.Linear(in_channels, out_channels, bias=self.bias)
        self.f_liner = nn.Linear(3 * out_channels, 1, bias=self.bias)
        self.sigmod = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        # 训练得到边的分数

    def forward(self, x, edge_index, edge_labels, edge_train_mask):
        x = self.d_liner(x)

        e_feats = torch.cat((x[edge_index[0]], x[edge_index[1]], (x[edge_index[0]] - x[edge_index[1]])), dim=1)
        e_feats = self.dropout(e_feats)
        edge_score = self.f_liner(e_feats).squeeze()
        edge_score = self.sigmod(edge_score)  # sigmod[0,1]

        # 得到边分类的损失
        edge_loss = self.edge_loss(edge_labels, edge_score, edge_train_mask)
        return edge_score, edge_loss

    def edge_loss(self, edge_labels, edge_score, edge_train_mask):
        # 使用焦点损失
       
        edge_loss = FocalLoss(gamma=self.gamma, alpha=self.alpha)(edge_score[edge_train_mask], edge_labels[edge_train_mask])
        return edge_loss

class Big_EC(torch.nn.Module):

    def __init__(self, in_channels, out_channels, alpha, gamma, dropout=0.1):
        super(Big_EC, self).__init__()
        self.bias = False
        self.alpha = alpha
        self.gamma = gamma
        self.d_liner = nn.Linear(in_channels, out_channels, bias=self.bias)
        self.f_liner = nn.Linear(3*out_channels, 1, bias=self.bias)
       
        self.sigmod = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        # 训练得到边的分数

    def forward(self, x, edge_index, edge_labels, edge_train_mask):
        x = self.d_liner(x)


        e_feats = torch.cat((x[edge_index[0]], x[edge_index[1]], (x[edge_index[0]] - x[edge_index[1]])), dim=1)  
        
        e_feats = self.dropout(e_feats)
        edge_score = self.f_liner(e_feats).squeeze()
        
        edge_score = self.sigmod(edge_score)  # sigmod[0,1]

        # 得到边分类的损失
        edge_loss = self.edge_loss(edge_labels, edge_score, edge_train_mask)
        return edge_score, edge_loss

    def edge_loss(self, edge_labels, edge_score, edge_train_mask):
        # 使用焦点损失
       
        edge_loss = FocalLoss(gamma=self.gamma, alpha=self.alpha)(edge_score[edge_train_mask], edge_labels[edge_train_mask])
        return edge_loss



class GroupGCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, beta, dropout):
        super(GroupGCN, self).__init__()
        self.bias = False
        self.cached = True
        self.self_loop = False
        self.beta = beta
        self.activation = nn.ReLU(inplace=True)
        self.dropout = dropout

        # 两层，每层三个聚合函数 分别处理自己、同质邻居和异质邻居
        self.center1 = nn.Linear(in_channels, hidden_channels, bias=self.bias)
        self.homo1 = GCNConv(in_channels, hidden_channels, cached=self.cached, bias=self.bias,
                             add_self_loops=self.self_loop)
        self.hetero1 = GCNConv(in_channels, hidden_channels, cached=self.cached, bias=self.bias,
                               add_self_loops=self.self_loop)

        self.center2 = nn.Linear(hidden_channels, out_channels, bias=self.bias)
        self.homo2 = GCNConv(hidden_channels, out_channels, cached=self.cached, bias=self.bias,
                             add_self_loops=self.self_loop)
        self.hetero2 = GCNConv(hidden_channels, out_channels, cached=self.cached, bias=self.bias,
                               add_self_loops=self.self_loop)

    def forward(self, x, homo_edge_index, hetero_edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)

        center1 = self.center1(x)
        h_homo = self.homo1(x, homo_edge_index)
        h_hetero = self.hetero1(x, hetero_edge_index)

        h_combined = center1 + self.beta*h_homo +(1-self.beta)*h_hetero

        h_combined = self.activation(h_combined)
        h_combined = F.dropout(h_combined, p=self.dropout, training=self.training)

        center2 = self.center2(h_combined)
        h_homo2 = self.homo2(h_combined, homo_edge_index)
        h_hetero2 = self.hetero2(h_combined, hetero_edge_index)
        h_final = center2 + self.beta*h_homo2 +(1-self.beta)* h_hetero2

        logits = F.log_softmax(h_final, dim=1)
        return F.softmax(h_final, dim=1), logits



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 同质边比例，异质边比例
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, logits, target):
        logits = torch.sigmoid(logits)
        logits = logits.view(-1)
        target = target.view(-1)
        
        logpt = torch.where(target == 1, torch.log(logits), torch.log(1 - logits))
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)

            at = self.alpha.gather(0, target.data.long())
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

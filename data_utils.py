import torch
import os
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork,Actor
import torch_geometric.transforms as T
from sklearn.metrics.pairwise import cosine_similarity as cos
from torch_geometric.utils import remove_self_loops,is_undirected, add_self_loops
import numpy as np
import scipy.sparse as sp
import random

# 基于随机游走扩散过程的结构编码来捕获结构信息。
def get_structural_encoding(edges, nnodes, str_enc_dim=16):
    edges = edges.cpu()
    row = edges[0, :].numpy()
    col = edges[1, :].numpy()
    data = np.ones_like(row)

    A = sp.csr_matrix((data, (row, col)), shape=(nnodes, nnodes))
    D = (np.array(A.sum(1)).squeeze()) ** -1.0

    Dinv = sp.diags(D)
    RW = A * Dinv
    M = RW

    SE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(str_enc_dim - 1):
        M_power = M_power * M
        SE.append(torch.from_numpy(M_power.diagonal()).float())
    SE = torch.stack(SE, dim=-1)
    return SE

# 返回边标签和掩码矩阵，同质边为1 异质边为0
def generate_edges_labels(edges, labels, train_idx):

    row, col = edges

    # 先转成索引
    train_idx = torch.nonzero(train_idx, as_tuple=True)[0]
    train_idx = set(train_idx.tolist())

    labels_row = labels[row]
    labels_col = labels[col]

    # 边的标签: 如果labels相同，则为1，否则为0
    edge_labels = (labels_row == labels_col).long()
    # 边的掩码: 如果i和j都在train_idx中，则为1，否则为0
    edge_train_mask = []
    for i, j in zip(row, col):
        i = i.item()
        j = j.item()
        if i in train_idx and j in train_idx:
            edge_train_mask.append(1)
        else:
            edge_train_mask.append(0)
    edge_train_mask = torch.Tensor(edge_train_mask).bool()
    if edge_train_mask.sum().item()==0:
        print("edge_train_mask=0")
#     print("edge_train_mask",edge_train_mask.sum())
    return edge_labels, edge_train_mask

# 虚拟边集合
'''
def generate_edge_trainset(labels, train_idx, n):
    # 构建正边和负边
    pos_edges = []
    neg_edges = []
    for node in train_idx.nonzero(as_tuple=True)[0]:
        # 获取当前节点的类别
        current_class = labels[node].item()

        # 获取同类节点和不同类节点的索引
        same_class_nodes = (labels == current_class).nonzero(as_tuple=True)[0]
        diff_class_nodes = (labels != current_class).nonzero(as_tuple=True)[0]

        # 排除自身
        same_class_nodes = same_class_nodes[same_class_nodes != node]

        # 随机选择一个同类节点构建正边
        if len(same_class_nodes) > 0:
            pos_nodes = random.choices(same_class_nodes.tolist(), k=n)
            pos_edges.extend([[node, pos_node] for pos_node in pos_nodes])

        # 随机选择一个不同类节点构建负边
        if len(diff_class_nodes) > 0:
            neg_nodes = random.choices(diff_class_nodes.tolist(), k=n)
            neg_edges.extend([[node, neg_node] for neg_node in neg_nodes])

    # 将正边和负边转换为张量并合并
    homo_edge_index = torch.tensor(pos_edges, dtype=torch.long).t().contiguous()
    hetero_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()

    return homo_edge_index, hetero_edge_index
'''
def calculate_classbalance(edge_labels):
    pos_count = (edge_labels == 1).sum().item()  # 同质边比例
    neg_count = (edge_labels == 0).sum().item()  # 异质边比例
    total_count = edge_labels.size(0)
    alpha_pos = neg_count / total_count  # 较大的alpha值给少类
    alpha_neg = pos_count / total_count  # 较小的alpha值给大类
    return torch.tensor([alpha_pos, alpha_neg])


def pre_edgeset(edges, labels, idx, device):
    edges = edges.cpu()
    labels = labels.cpu()
    idx = idx.cpu()
    
    edge_labels, edge_train_mask = generate_edges_labels(edges=edges, labels=labels, train_idx=idx)
    new_edges = edges.to(device)
    new_edge_labels = edge_labels.to(device)
    edge_train_mask = edge_train_mask.to(device)
    alpha = calculate_classbalance(new_edge_labels)
    return new_edges, new_edge_labels, edge_train_mask, alpha


# knn图的创建
def adj_knn(dataset):
    features = dataset.x

    for topk in range(2, 10):
        print(topk)
        construct_graph(dataset, features, topk)
        f1 = open('./new_pyg_data/' + dataset + '/knn/tmp.txt','r')
        f2 = open('./new_pyg_data/' + dataset + '/knn/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()

def construct_graph(dataset, features, topk):
    fname = './new_pyg_data/' + dataset + '/knn/tmp.txt'
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            pass  # 创建一个空文件

    f = open(fname, 'w')
    # Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    # Cosine 余弦相似度建立knn
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()

# LINKS大数据集的..
def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


# def to_sparse_tensor(edge_index, edge_feat, num_nodes):
#     """ converts the edge_index into SparseTensor
#     """
#     num_edges = edge_index.size(1)
#
#     (row, col), N, E = edge_index, num_nodes, num_edges
#     perm = (col * N + row).argsort()
#     row, col = row[perm], col[perm]
#
#     value = edge_feat[perm]
#     adj_t = SparseTensor(row=col, col=row, value=value,
#                          sparse_sizes=(N, N), is_sorted=True)
#
#     # Pre-process some important attributes.
#     adj_t.storage.rowptr()
#     adj_t.storage.csr2csc()
#
#     return adj_t

def rand_train_test_idx(label, train_prop=.48, valid_prop=.32, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num].long()
    val_indices = perm[train_num:train_num + valid_num].long()
    test_indices = perm[train_num + valid_num:].long()

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx
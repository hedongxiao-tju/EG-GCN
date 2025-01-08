import warnings
import torch

warnings.filterwarnings('ignore')
import copy
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import is_undirected, to_undirected, one_hot
from utils import *
from data_utils import *
from data_loadder import load_data
from model import *
from collections import Counter
import argparse
import numpy as np
import time
from scipy.sparse import coo_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="wisconsin",
                    choices=["Cora", "Pubmed", "Citeseer", "Actor", "texas", "wisconsin", "cornell", "chameleon",
                             "squirrel"])
parser.add_argument('--lr_ec', type=float, default=0.05, help='Learning rate for edge classify.')
parser.add_argument('--lr_gcn', type=float, default=0.05, help='Learning rate for gcn.')
parser.add_argument('--decay_ec', type=float, default=5e-4, help='Weight decay for edge classify optimization.')
parser.add_argument('--decay_gcn', type=float, default=5e-4, help='Weight decay for gcn optimization.')
parser.add_argument('--epochs_ec', type=int, default=100, help='Number of training epochs per iteration.')
parser.add_argument('--epochs_gcn', type=int, default=300, help='Number of training epochs per iteration.')
parser.add_argument('--hdim_ec', type=int, default=256, help='Hidden embedding dimension.')
parser.add_argument('--hdim_gcn', type=int, default=256, help='Hidden embedding dimension.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout ratio.')
parser.add_argument('--coiters', type=int, default=4, help='Co-training iters.')
parser.add_argument('--confidence', type=float, default=0.99, help='High confidence for pseudo labels  .')
parser.add_argument('--beta', type=float, default=0.5, help='Combine hyper-parameters.')
parser.add_argument('--gamma', type=int, default=2, help='Focal Loss hyper-parameters.')
parser.add_argument('--seed', type=int, default=1262)
parser.add_argument('--split', type=int, default=10, help='Total splits.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    print("CUDA EXISTS!")
    device = "cuda:0"
    torch.cuda.manual_seed(args.seed)
else:
    device = "cpu"

# 加载数据集
dataset = load_data(data=args.dataset)

num_nodes = dataset[0].x.size(0)
num_features = dataset.num_features
num_classes = dataset.num_classes
data = dataset[0]
features = data.x.to(device)
labels = data.y.to(device)
edges = data.edge_index.to(device)

# 变成无向无环图进行训练边分类器
edges, _ = remove_self_loops(edges)
if not is_undirected(edges):
    # 如果不是无向图，添加反向边使其变为无向图
    edges = to_undirected(edges)
    print("Converted to undirected:", edges.size())
else:
    print("Already undirected.", edges.size())
edges = edges.to(device)
print(data)

if args.dataset in ['genius', 'Penn94']:
    print("No use SE!")
else:
    print("Use SE!")
    # 结构编码
    path = './struct_encoding/{}'.format(args.dataset)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + '/{}_{}.pt'.format(args.dataset, 16)
    if os.path.exists(file_name):
        strcutural_encoding = torch.load(file_name)
        print('Load exist structural encoding.')
    else:
        print('Computing structural encoding...')
        strcutural_encoding = get_structural_encoding(edges, num_nodes)
        torch.save(strcutural_encoding, file_name)
        print('Done. The structural encoding is saved as: {}.'.format(file_name))
    strcutural_encoding = strcutural_encoding.to(device)


def train_edgeclassfy(model, input, new_edges, edge_labels, edge_train_mask, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_ec, weight_decay=args.decay_ec)

    best_epoch, best_acc_edge = -1, 0
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        edge_score, edge_loss = model(x=input, edge_index=new_edges, edge_labels=edge_labels,
                                      edge_train_mask=edge_train_mask)

        edge_loss.backward(retain_graph=True)
        optimizer.step()
        model.eval()
        acc_edge = edge_accuracy(edge_score[~edge_train_mask], edge_labels[~edge_train_mask])

        if acc_edge > best_acc_edge:
            best_epoch = i
            best_acc_edge = acc_edge

    print('Best EC Acc: {:.4f}'.format(best_acc_edge))
    return model  # 返回模型以供下次继续训练


def train_groupgcn(model, epochs, homo_edge_index, hetero_edge_index, idx_train, idx_val, idx_test):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_gcn, weight_decay=args.decay_gcn)
    best_epoch, best_val_acc, best_test_acc, best_f1 = -1, 0, 0, 0
    best_pred = []
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds, logits = model(x=features, homo_edge_index=homo_edge_index, hetero_edge_index=hetero_edge_index)
        loss_train = F.nll_loss(logits[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        acc_val = accuracy(logits[idx_val], labels[idx_val])
        if acc_val > best_val_acc:
            best_epoch = i
            best_val_acc = acc_val
            best_test_acc = accuracy(logits[idx_test], labels[idx_test])
            best_pred = preds
            best_f1 = getf1(out=logits[idx_test], labels=labels[idx_test])

    print('{}|'.format(args.dataset),
          'e:{}|'.format(best_epoch + 1),
          'BEST Val Acc: {:.4f}|'.format(best_val_acc.item()),
          'BEST Test Acc: {:.4f}|'.format(best_test_acc.item()),
          'BEST Test F1:{:.4f}'.format(best_f1))

    confidence_threshold = args.confidence
    max_probs, pseudo_labels = torch.max(best_pred[~idx_train], dim=1)
    if max_probs.size(0) > 50:
        selected_indices = torch.randperm(max_probs.size(0))[:50]
        max_probs = max_probs[selected_indices]
        pseudo_labels = pseudo_labels[selected_indices]

    high_confidence_mask = max_probs >= confidence_threshold
    high_confidence_pseudo_labels = pseudo_labels[high_confidence_mask]
    high_confidence_indices = torch.nonzero(high_confidence_mask).squeeze()
    print("High confidence pseudo labels num:", high_confidence_indices.size())
    return model, best_test_acc, best_f1, high_confidence_pseudo_labels, high_confidence_indices

def run(idx_train, idx_val, idx_test):
    pseudo_labels = torch.zeros(labels.size(), dtype=torch.long).to(device)
    pseudo_labels.copy_(labels)
    pseudo_trainidx = torch.zeros(idx_train.size(), dtype=torch.bool).to(device)
    pseudo_trainidx.copy_(idx_train)

    best_test_acc, best_test_f1 = 0, 0

    # 初始化模型
    EC_model = None
    GCN_model = None
    edge_input = torch.cat((features, strcutural_encoding), dim=1)
    for i in range(args.coiters):
        new_edges, new_edge_labels, edge_train_mask, alpha = pre_edgeset(edges=edges, labels=pseudo_labels,
                                                                         idx=pseudo_trainidx, device=device)
        if EC_model is None:
            EC_model = EC(in_channels=edge_input.size(1), out_channels=args.hdim_ec,
                          alpha=alpha, gamma=args.gamma, dropout=args.dropout).to(device)

        EC_model = train_edgeclassfy(EC_model, edge_input, new_edges, new_edge_labels, edge_train_mask,
                                     epochs=args.epochs_ec)

        edge_score, _ = EC_model(x=edge_input, edge_index=edges,
                                 edge_labels=new_edge_labels[:edges.size(1)],
                                 edge_train_mask=edge_train_mask[:edges.size(1)])

        homo_mask = (edge_score > 0.5) | ((edge_train_mask[:edges.size(1)]) & (new_edge_labels[:edges.size(1)] == 1))
        hetero_mask = ~homo_mask  
        homo_edge_index = edges[:, homo_mask]
        hetero_edge_index = edges[:, hetero_mask]

        if GCN_model is None:
            GCN_model = GroupGCN(in_channels=num_features, hidden_channels=args.hdim_gcn, out_channels=num_classes,
                                 beta=args.beta, dropout=args.dropout).to(device)

        GCN_model, test_acc, test_f1, high_confidence_pseudo_labels, high_confidence_indices = train_groupgcn(GCN_model,
                                                                                                              epochs=args.epochs_gcn,
                                                                                                              homo_edge_index=homo_edge_index,
                                                                                                              hetero_edge_index=hetero_edge_index,
                                                                                                              idx_train=idx_train,
                                                                                                              idx_val=idx_val,
                                                                                                              idx_test=idx_test)

        pseudo_labels[high_confidence_indices] = high_confidence_pseudo_labels  # 更新高置信度伪标签
        pseudo_trainidx[high_confidence_indices] = True  # 将高置信度样本标记为训练样本

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = test_f1
#         best_test_acc = test_acc
    print('Best GCN Acc: {:.4f}| Best GCN F1: {:.4f}'.format(best_test_acc, best_test_f1))
    return best_test_acc, best_test_f1


if __name__ == "__main__":
    all_acc = []
    all_f1 = []
    for split in range(args.split):
        print("================================================================")
        print(f"Running split {split}..." + '\n')
        data.train_mask = data.train_mask.bool()
        data.val_mask = data.val_mask.bool()
        data.test_mask = data.test_mask.bool()
        idx_train = data.train_mask[:, split].to(device)
        idx_val = data.val_mask[:, split].to(device)
        idx_test = data.test_mask[:, split].to(device)

        start_time = time.time()
        acc, f1 = run(idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
        end_time = time.time()
        print(f"One Split time: {end_time - start_time} seconds")

        all_acc.append(acc.cpu())
        all_f1.append(f1)

    all_acc = np.array(all_acc)
    all_f1 = np.array(all_f1)
    print("ALL FINISH!")
    print("All results:", all_acc)
    print('Avg acc: {:.2f}±{:.2f}'.format(all_acc.mean() * 100, all_acc.std() * 100))
    print('Avg f1: {:.2f}±{:.2f}'.format(all_f1.mean() * 100, all_f1.std() * 100))
    print(args)
#     with open('./log/{}.txt'.format(args.dataset), 'a') as f:
#         f.write('\n\n' + '##' * 20 + '\n')
#         f.write('All results: {}\n'.format(all_acc))
#         f.write('Avg acc: {:.2f}±{:.2f}\n'.format(all_acc.mean() * 100, all_acc.std() * 100))
#         f.write('Avg f1: {:.2f}±{:.2f}'.format(all_f1.mean() * 100, all_f1.std() * 100) + '\n')
#         f.write(str(args) + '\n')
    with open('./log/param{}.txt'.format(args.dataset), 'a') as f:
        f.write(' coiters{:} confidence{:.2f}  '.format(args.coiters,args.confidence))
#         f.write('beta{:.1f}  coiters{:} '.format(args.beta, args.coiters))
        f.write('Avg acc: {:.2f}\n'.format(all_acc.mean() * 100))

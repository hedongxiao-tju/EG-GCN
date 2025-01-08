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
from scipy.sparse.linalg import svds

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="genius",
                    choices=['genius', 'Penn94'])
parser.add_argument('--lr_ec', type=float, default=0.001, help='Learning rate for edge classify.')
parser.add_argument('--lr_gcn', type=float, default=0.1, help='Learning rate for gcn.')
parser.add_argument('--decay_ec', type=float, default=5e-4, help='Weight decay for edge classify optimization.')
parser.add_argument('--decay_gcn', type=float, default=5e-4, help='Weight decay for gcn optimization.')
parser.add_argument('--epochs_ec', type=int, default=50, help='Number of training epochs per iteration.')
parser.add_argument('--epochs_gcn', type=int, default=50, help='Number of training epochs per iteration.')
parser.add_argument('--split', type=int, default=1, help='Total splits.')
parser.add_argument('--hdim_ec', type=int, default=8, help='Hidden embedding dimension.')
parser.add_argument('--hdim_gcn', type=int, default=16, help='Hidden embedding dimension.')
parser.add_argument('--dropout', type=float, default=0.05, help='Dropout ratio.')
parser.add_argument('--coiters', type=int, default=10, help='Co-training iters.')
parser.add_argument('--confidence', type=float, default=0.97, help='High confidence for pseudo labels  .')
parser.add_argument('--beta', type=float, default=0.5, help='GCN hyper-parameters.')
parser.add_argument('--gamma', type=int, default=2, help='Focal Loss hyper-parameters.')
parser.add_argument('--seed', type=int, default=1262)
args = parser.parse_args()

# 设置随机数种子和设备
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    print("CUDAS!")
    device = "cuda:0"
    torch.cuda.manual_seed(args.seed)
else:
    device = "cpu"

# 加载数据集
dataset = load_data(data=args.dataset)

num_nodes = dataset.x.size(0)
num_features = dataset.x.size(1)
data = dataset
features = data.x.to(device)
#features = F.normalize(data.x.to(device), p=2, dim=-1)
labels = data.y.to(torch.int64).to(device)  # Penn94 gender label,-1 means unlabeled
edges = data.edge_index.to(device)
print(data)
if args.dataset == "genius":
    # genius需要特别处理，使用BCE交叉熵损失函数和AUC指标。
    num_classes = dataset.num_class
    if len(labels.shape) == 1:
        true_label = labels.unsqueeze(1)
    if true_label.shape[1] == 1:
        true_label = F.one_hot(true_label, true_label.max() + 1).squeeze(1).to(device)
    else:
        true_label = true_label.to(device)
    criterion = nn.BCEWithLogitsLoss()
else:
    num_classes = dataset.num_classes
    criterion = nn.NLLLoss()


def train_edgeclassfy(model, new_edges, edge_labels, edge_train_mask, alpha, epochs):

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_ec, weight_decay=args.decay_ec)

    best_epoch, best_acc_edge = -1, 0
    print("Train Edge..")
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        edge_score, edge_loss = model(x=features, edge_index=new_edges, edge_labels=edge_labels,
                                      edge_train_mask=edge_train_mask)

        edge_loss.backward()
        optimizer.step()
        model.eval()
        # 边分类器的准确率
        acc_edge = edge_accuracy(edge_score[~edge_train_mask], edge_labels[~edge_train_mask])

        if acc_edge > best_acc_edge:
            best_epoch = i
            best_acc_edge = acc_edge
        if i % 10 == 0:
            print('e:{}'.format(i + 1),
                  'Train Loss: {:.4f}'.format(edge_loss.item()),
                  'Edge Acc: {:.4f}'.format(acc_edge))
    print('Best EC Acc: {:.4f}'.format(best_acc_edge))
    return model

def train_groupgcn(model,epochs, homo_edge_index, hetero_edge_index, idx_train, idx_val, idx_test):

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_gcn, weight_decay=args.decay_gcn)

    best_epoch, best_val_acc, best_test_acc, best_f1 = -1, 0, 0, 0
    best_pred = []
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds, logits = model(x=features, homo_edge_index=homo_edge_index, hetero_edge_index=hetero_edge_index)
        if args.dataset == "genius":
            loss_train = criterion(logits[idx_train], true_label.squeeze(1)[idx_train].to(torch.float))
        else:
            loss_train = criterion(logits[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        model.eval()
        acc_train = accuracy(logits[idx_train], labels[idx_train])

        if args.dataset == "genius":
            acc_val = eval_rocauc(preds[idx_val], true_label[idx_val])
        else:
            acc_val = accuracy(logits[idx_val], labels[idx_val])
        if acc_val > best_val_acc:
            best_epoch = i
            best_val_acc = acc_val
            best_pred = preds
            best_f1 = getf1(out=logits[idx_test], labels=labels[idx_test])
            if args.dataset == "genius":
                best_test_acc = eval_rocauc(preds[idx_test], true_label[idx_test])
            else:
                best_test_acc = accuracy(logits[idx_test], labels[idx_test])

        if i % 50 == 0:
            print('e:{}'.format(i + 1),
                  'Train Loss: {:.4f}'.format(loss_train.item()),
                  'Train Acc: {:.4f}'.format(acc_train.item()),
                  'Val Acc: {:.4f}'.format(acc_val.item()))

    print('{}|'.format(args.dataset),
          'e:{}|'.format(best_epoch + 1),
          'BEST Val Acc: {:.4f}|'.format(best_val_acc.item()),
          'BEST Test Acc: {:.4f}|'.format(best_test_acc.item()),
          'BEST Test F1:{:.4f}'.format(best_f1))

    # 获得高置信度的伪标签
    confidence_threshold = args.confidence
    max_probs, pseudo_labels = torch.max(best_pred[~idx_train], dim=1)
    if max_probs.size(0) > 50:
        selected_indices = torch.randperm(max_probs.size(0))[:50]
        max_probs = max_probs[selected_indices]
        pseudo_labels = pseudo_labels[selected_indices]

    # 筛选出测试集和验证集中置信度高的伪标签
    high_confidence_mask = max_probs >= confidence_threshold
    high_confidence_pseudo_labels = pseudo_labels[high_confidence_mask]
    # 获取高置信度样本的索引
    high_confidence_indices = torch.nonzero(high_confidence_mask).squeeze()
    # 输出高置信度伪标签及其索引
    print("High confidence pseudo labels num:", high_confidence_indices.size())
    return model, best_test_acc, best_f1, high_confidence_pseudo_labels, high_confidence_indices


def run(idx_train, idx_val, idx_test):
    pseudo_labels = torch.zeros(labels.size(), dtype=torch.long).to(device)
    pseudo_labels.copy_(labels)
    pseudo_trainidx = torch.zeros(idx_train.size(), dtype=torch.bool).to(device)
    pseudo_trainidx.copy_(idx_train)

    best_test_acc = 0
    EC_model = None
    GCN_model = None
    for i in range(args.coiters):
        print("Edge Start..")
        #         print(pseudo_labels.shape)
        new_edges, new_edge_labels, edge_train_mask, alpha = pre_edgeset(edges=edges, labels=pseudo_labels,
                                                                         idx=pseudo_trainidx,device=device)
        print(edge_train_mask.sum())
        print(alpha)
        if EC_model is None:
            EC_model = Big_EC(in_channels=features.size(1), out_channels=args.hdim_ec, alpha=alpha, gamma=args.gamma,
                       dropout=args.dropout).to(device)
          
        EC_model = train_edgeclassfy(EC_model, new_edges, new_edge_labels, edge_train_mask, alpha, epochs=args.epochs_ec)

        EC_model.eval()
        edge_score, _ = EC_model(x=features, edge_index=edges, edge_labels=new_edge_labels[:edges.size(1)],
                                 edge_train_mask=edge_train_mask[:edges.size(1)])
        
#         edge_score = new_edge_labels

        homo_mask = (edge_score > 0.5) | ((edge_train_mask[:edges.size(1)]) & (new_edge_labels[:edges.size(1)] == 1))
        hetero_mask = ~homo_mask  # 取homo_mask的反面即可
        homo_edge_index = edges[:, homo_mask]
        hetero_edge_index = edges[:, hetero_mask]

        if GCN_model is None:
            GCN_model = GroupGCN(in_channels=num_features, hidden_channels=args.hdim_gcn, out_channels=num_classes,
                         beta=args.beta,
                         dropout=args.dropout).to(device)

        GCN_model, test_acc, test_f1, high_confidence_pseudo_labels, high_confidence_indices = train_groupgcn(model=GCN_model,
            epochs=args.epochs_gcn, homo_edge_index=homo_edge_index, hetero_edge_index=hetero_edge_index,
            idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

        pseudo_labels[high_confidence_indices] = high_confidence_pseudo_labels  # 更新高置信度伪标签
        pseudo_trainidx[high_confidence_indices] = True  # 将高置信度样本标记为训练样本
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = test_f1
    print('Best GCN Acc: {:.4f}| Best GCN F1: {:.4f}'.format(best_test_acc, best_test_f1))
    return best_test_acc, best_test_f1


if __name__ == "__main__":
    all_acc, all_f1 = [], []
    for split in range(args.split):
        print("================================================================")
        print(f"Running split {split}..." + '\n')
        if args.dataset in ['genius', 'Penn94']:
            masks = np.load(f"./data/{args.dataset}_masks.npz")
            idx_train = torch.tensor(masks["train_mask"][:, split]).to(device)
            idx_val = torch.tensor(masks["valid_mask"][:, split]).to(device)
            idx_test = torch.tensor(masks["test_mask"][:, split]).to(device)
        start_time = time.time()
        acc, f1 = run(idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
        end_time = time.time()
        print(f"One Split time: {end_time - start_time} seconds")

        if args.dataset == "genius":
            all_acc.append(acc)
            all_f1.append(f1)
        else:
            all_acc.append(acc.cpu())
            all_f1.append(f1)

    all_acc = np.array(all_acc)
    all_f1 = np.array(all_f1)
    print("ALL FINISH!")
    print("All results:", all_acc)
    print('Avg acc: {:.2f}±{:.2f}'.format(all_acc.mean() * 100, all_acc.std() * 100))
    print('Avg f1: {:.2f}±{:.2f}'.format(all_f1.mean() * 100, all_f1.std() * 100))
    print(args)
    with open('./log/{}.txt'.format(args.dataset), 'a') as f:
        f.write('\n\n' + '##' * 20 + '\n')
        f.write('All results: {}\n'.format(all_acc))
        f.write('Avg acc: {:.2f}±{:.2f}\n'.format(all_acc.mean() * 100, all_acc.std() * 100))
        f.write('Avg f1: {:.2f}±{:.2f}'.format(all_f1.mean() * 100, all_f1.std() * 100) + '\n')
        f.write(str(args) + '\n')


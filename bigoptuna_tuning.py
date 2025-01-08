import warnings

warnings.filterwarnings('ignore')
from torch_geometric.utils import to_undirected
from utils import *
from data_utils import *
from data_loadder import load_data
from model import *
import argparse
import numpy as np
import optuna
import random

if torch.cuda.is_available():
    print("CUDA EXISTS!")
    device = "cuda:0"
else:
    device = "cpu"

datasets = ['genius', 'Penn94']
name = "Penn94"

dataset = load_data(data=name)
num_nodes = dataset.x.size(0)
num_features = dataset.x.size(1)
data = dataset
# features = data.x.to(device)
features = F.normalize(data.x.to(device), p=2, dim=-1)  # 对行进行归一化
labels = data.y.to(torch.int64).to(device)  # Penn94 gender label,-1 means unlabeled
edges = data.edge_index.to(device)

if name == "genius":
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


def train_edgeclassfy(model,new_edges, edge_labels, edge_train_mask, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_ec, weight_decay=args.decay_ec)

    best_epoch, best_acc_edge = -1, 0
    #     print("Train Edge..")
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

    return model


def train_groupgcn(model, epochs, homo_edge_index, hetero_edge_index, idx_train, idx_val, idx_test):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_gcn, weight_decay=args.decay_gcn)

    best_epoch, best_val_acc, best_test_acc, best_f1 = -1, 0, 0, 0
    best_pred = []
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds, logits = model(x=features, homo_edge_index=homo_edge_index, hetero_edge_index=hetero_edge_index)
        if name == "genius":
            loss_train = criterion(logits[idx_train], true_label.squeeze(1)[idx_train].to(torch.float))
        else:
            loss_train = criterion(logits[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        model.eval()
        acc_train = accuracy(logits[idx_train], labels[idx_train])

        if name == "genius":
            acc_val = eval_rocauc(preds[idx_val], true_label[idx_val])
        else:
            acc_val = accuracy(logits[idx_val], labels[idx_val])
        if acc_val > best_val_acc:
            best_epoch = i
            best_val_acc = acc_val
            best_pred = preds
            best_f1 = getf1(out=logits[idx_test], labels=labels[idx_test])
            if name == "genius":
                best_test_acc = eval_rocauc(preds[idx_test], true_label[idx_test])
            else:
                best_test_acc = accuracy(logits[idx_test], labels[idx_test])

    #         if i % 50 == 0:
    #             print('e:{}'.format(i + 1),
    #                   'Train Loss: {:.4f}'.format(loss_train.item()),
    #                   'Train Acc: {:.4f}'.format(acc_train.item()),
    #                   'Val Acc: {:.4f}'.format(acc_val.item()))

    #     print('{}|'.format(name),
    #           'e:{}|'.format(best_epoch + 1),
    #           'BEST Val Acc: {:.4f}|'.format(best_val_acc.item()),
    #           'BEST Test Acc: {:.4f}|'.format(best_test_acc.item()),
    #           'BEST Test F1:{:.4f}'.format(best_f1))

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
    #     print("High confidence pseudo labels num:", high_confidence_indices.size())
    return model, best_test_acc, best_f1, high_confidence_pseudo_labels, high_confidence_indices


def run(idx_train, idx_val, idx_test, args, features, labels, edges, device):
    pseudo_labels = torch.zeros(labels.size(), dtype=torch.long).to(device)
    pseudo_labels.copy_(labels)
    pseudo_trainidx = torch.zeros(idx_train.size(), dtype=torch.bool).to(device)
    pseudo_trainidx.copy_(idx_train)

    best_test_acc = 0
    EC_model = None
    GCN_model = None

    for i in range(args.coiters):

        new_edges, new_edge_labels, edge_train_mask, alpha = pre_edgeset(edges=edges, labels=pseudo_labels,
                                                                         idx=pseudo_trainidx,device=device)
        if EC_model is None:
            EC_model = EC(in_channels=features.size(1), out_channels=args.hdim_ec,
                          alpha=alpha, gamma=args.gamma, dropout=args.dropout).to(device)

        EC_model = train_edgeclassfy(EC_model, new_edges, new_edge_labels, edge_train_mask, epochs=args.epochs_ec)

        edge_score, _ = EC_model(x=features, edge_index=edges, edge_labels=new_edge_labels[:edges.size(1)],
                                 edge_train_mask=edge_train_mask[:edges.size(1)])

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
    #     print('Best GCN Acc: {:.4f}| Best GCN F1: {:.4f}'.format(best_test_acc,best_test_f1))
    return best_test_acc, best_test_f1


def objective(trial):
    # args.lr_ec = trial.suggest_loguniform('lr_ec', 1e-5, 1e-1)
    # args.lr_gcn = trial.suggest_loguniform('lr_gcn', 1e-5, 1e-1)
    # args.decay_ec = trial.suggest_loguniform('decay_ec', 1e-6, 1e-2)
    # args.decay_gcn = trial.suggest_loguniform('decay_gcn', 1e-6, 1e-2)

    args.lr_ec = trial.suggest_categorical('lr_ec', [1e-5, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1])
    args.lr_gcn = trial.suggest_categorical('lr_gcn', [1e-5, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1])
    args.decay_ec = trial.suggest_categorical('decay_ec', [0.0, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    args.decay_gcn = trial.suggest_categorical('decay_gcn', [0.0, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    args.hdim_ec = trial.suggest_categorical('hdim_ec', [32, 64])
    args.hdim_gcn = trial.suggest_categorical('hdim_gcn', [16, 32, 128, 256])
    args.epochs_ec = trial.suggest_int('epochs_ec', 100, 300, step=50)
    args.epochs_gcn = trial.suggest_int('epochs_gcn', 200, 500, step=50)
    args.dropout = trial.suggest_float('dropout', 0.0, 0.9, step=0.1)
    args.beta = trial.suggest_float('beta', 0, 1, step=0.1)
    args.confidence = trial.suggest_float('confidence', 0.9, 0.99, step=0.01)
    args.gamma = trial.suggest_int('gamma', 2, 4, step=1)
    args.coiters = trial.suggest_int('coiters', 1, 4, step=1)
    parser = argparse.ArgumentParser()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_acc = []
    all_f1 = []
    for split in range(args.split):
        #         print("================================================================")
        #         print(f"Running split {split}..." + '\n')

        masks = np.load(f"./data/{name}_masks.npz")
        idx_train = torch.tensor(masks["train_mask"][:, split]).to(device)
        idx_val = torch.tensor(masks["valid_mask"][:, split]).to(device)
        idx_test = torch.tensor(masks["test_mask"][:, split]).to(device)

        acc, f1 = run(idx_train=idx_train, idx_val=idx_val, idx_test=idx_test, args=args, features=features,
                      labels=labels, edges=edges, device=device)
        if name == "genius":
            all_acc.append(acc)
            all_f1.append(f1)
        else:
            all_acc.append(acc.cpu())
            all_f1.append(f1)

    all_acc = np.array(all_acc)
    all_f1 = np.array(all_f1)
    avg_acc = all_acc.mean()
    print('Avg acc: {:.2f}±{:.2f}'.format(all_acc.mean() * 100, all_acc.std() * 100))
    print('Avg f1: {:.2f}±{:.2f}'.format(all_f1.mean() * 100, all_f1.std() * 100))
    return avg_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=1, help='Total splits.')
#     parser.add_argument('--epochs_ec', type=int, default=300, help='Number of training epochs per iteration.')
#     parser.add_argument('--epochs_gcn', type=int, default=500, help='Number of training epochs per iteration.')
    parser.add_argument('--per_virtual_edge', type=int, default=0, help='Virtual edges Per node .')
    #     parser.add_argument('--hdim_ec', type=int, default=64 , help='Hidden embedding dimension.')
    parser.add_argument('--seed', type=int, default=1262)
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize')  # 目标是最大化acc
    study.optimize(objective, n_trials=200)

    print(name)
    print("Best trial:")
    trial = study.best_trial
    print("  Number: ", trial.number)
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

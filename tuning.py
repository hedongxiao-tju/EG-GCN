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

datasets = ["Cora", "Pubmed", "Citeseer", "Actor", "texas", "wisconsin", "cornell", "chameleon",
            "squirrel", 'genius', 'Penn94']
name = "Cora"
dataset = load_data(data=name)

num_nodes = dataset[0].x.size(0)
num_features = dataset.num_features
num_classes = dataset.num_classes
data = dataset[0]
features = data.x.to(device)
labels = data.y.to(device)
edges = data.edge_index.to(device)

edges, _ = remove_self_loops(edges)
if not is_undirected(edges):
    edges = to_undirected(edges)
    print("Converted to undirected:", edges.size())
else:
    print("Already undirected.", edges.size())
edges = edges.to(device)
print(data)

path = './struct_encoding/{}'.format(name)
if not os.path.exists(path):
    os.makedirs(path)
file_name = path + '/{}_{}.pt'.format(name, 16)
if os.path.exists(file_name):
    strcutural_encoding = torch.load(file_name)
    print('Load exist structural encoding.')
else:
    print('Computing structural encoding...')
    strcutural_encoding = get_structural_encoding(edges, num_nodes)
    torch.save(strcutural_encoding, file_name)
    print('Done. The structural encoding is saved as: {}.'.format(file_name))
strcutural_encoding = strcutural_encoding.to(device)

def train_edgeclassfy(model,input, new_edges, edge_labels, edge_train_mask, epochs, args,
                      device):

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

    return model


def train_groupgcn(model, epochs, homo_edge_index, hetero_edge_index, idx_train, idx_val, idx_test, args, features, labels,
                   num_features, num_classes, device):

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

    confidence_threshold = args.confidence
    max_probs, pseudo_labels = torch.max(best_pred[~idx_train], dim=1)
    if max_probs.size(0) > 50:
        selected_indices = torch.randperm(max_probs.size(0))[:50]
        max_probs = max_probs[selected_indices]
        pseudo_labels = pseudo_labels[selected_indices]

    high_confidence_mask = max_probs >= confidence_threshold
    high_confidence_pseudo_labels = pseudo_labels[high_confidence_mask]
    high_confidence_indices = torch.nonzero(high_confidence_mask).squeeze()
    return model, best_test_acc, best_f1, high_confidence_pseudo_labels, high_confidence_indices


def run(idx_train, idx_val, idx_test, args, features, labels, edges, strcutural_encoding, device):
    pseudo_labels = torch.zeros(labels.size(), dtype=torch.long).to(device)
    pseudo_labels.copy_(labels)
    pseudo_trainidx = torch.zeros(idx_train.size(), dtype=torch.bool).to(device)
    pseudo_trainidx.copy_(idx_train)

    best_test_acc, best_test_f1 = 0, 0
    EC_model = None
    GCN_model = None
    edge_input = torch.cat((features, strcutural_encoding), dim=1)
    for i in range(args.coiters):
        new_edges, new_edge_labels, edge_train_mask, alpha = pre_edgeset(edges=edges, labels=pseudo_labels,
                                                                         idx=pseudo_trainidx, device=device)

        if EC_model is None:
            EC_model = EC(in_channels=edge_input.size(1), out_channels=args.hdim_ec,
                          alpha=alpha, gamma=args.gamma, dropout=args.dropout).to(device)

        EC_model = train_edgeclassfy(EC_model, edge_input, new_edges, new_edge_labels, edge_train_mask, epochs=args.epochs_ec, args=args,device=device)

        edge_score, _ = EC_model(x=edge_input, edge_index=edges, edge_labels=new_edge_labels[:edges.size(1)],
                                 edge_train_mask=edge_train_mask[:edges.size(1)])

        homo_mask = (edge_score > 0.5) | ((edge_train_mask[:edges.size(1)]) & (new_edge_labels[:edges.size(1)] == 1))
        hetero_mask = ~homo_mask  # 取homo_mask的反面即可
        # homo_mask = edge_score > 0.5
        # hetero_mask = edge_score <= 0.5
        homo_edge_index = edges[:, homo_mask]
        hetero_edge_index = edges[:, hetero_mask]

        if GCN_model is None:
            GCN_model = GroupGCN(in_channels=num_features, hidden_channels=args.hdim_gcn, out_channels=num_classes,
                         beta=args.beta,
                         dropout=args.dropout).to(device)

        GCN_model, test_acc, test_f1, high_confidence_pseudo_labels, high_confidence_indices = train_groupgcn(model=GCN_model,
            epochs=args.epochs_gcn, homo_edge_index=homo_edge_index, hetero_edge_index=hetero_edge_index,
            idx_train=idx_train, idx_val=idx_val, idx_test=idx_test, args=args, features=features, labels=labels,
            num_features=num_features, num_classes=num_classes, device=device)

        pseudo_labels[high_confidence_indices] = high_confidence_pseudo_labels
        pseudo_trainidx[high_confidence_indices] = True
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = test_f1
#         print('Best GCN Acc: {:.4f}| Best GCN F1: {:.4f}'.format(best_test_acc,best_test_f1))
    return best_test_acc, best_test_f1


def objective(trial):

    args.lr_ec = trial.suggest_categorical('lr_ec', [0.001, 0.005, 0.01, 0.05, 0.1])
    args.lr_gcn = trial.suggest_categorical('lr_gcn', [0.001, 0.005, 0.01, 0.05, 0.1])
    args.decay_ec = trial.suggest_categorical('decay_ec', [0.0, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    args.decay_gcn = trial.suggest_categorical('decay_gcn', [0.0, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    args.hdim_ec = trial.suggest_categorical('hdim_ec', [64, 128, 256])
    args.hdim_gcn = trial.suggest_categorical('hdim_gcn', [64, 128, 256, 512, 768])
    args.dropout = trial.suggest_float('dropout', 0.0, 0.9, step=0.1)
    args.beta = trial.suggest_float('beta', 0.1, 1, step=0.1)
    args.confidence = trial.suggest_float('confidence', 0.81, 0.99, step=0.02)
    args.gamma = trial.suggest_int('gamma', 2, 4, step=1)
    args.coiters = trial.suggest_int('coiters', 1, 5, step=1)
    args.epochs_ec = trial.suggest_int('epochs_ec', 100, 300, step=50)
    args.epochs_gcn = trial.suggest_int('epochs_gcn', 200, 500, step=50)
    parser = argparse.ArgumentParser()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_acc = []
    all_f1 = []
    for split in range(args.split):
        #         print("================================================================")
        #         print(f"Running split {split}..." + '\n')
        data.train_mask = data.train_mask.bool()
        data.val_mask = data.val_mask.bool()
        data.test_mask = data.test_mask.bool()
        idx_train = data.train_mask[:, split].to(device)
        idx_val = data.val_mask[:, split].to(device)
        idx_test = data.test_mask[:, split].to(device)

        acc, f1 = run(idx_train=idx_train, idx_val=idx_val, idx_test=idx_test, args=args, features=features,
                      labels=labels, edges=edges, strcutural_encoding=strcutural_encoding, device=device)
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
#     parser.add_argument('--epochs_ec', type=int, default=200, help='Number of training epochs per iteration.')
#     parser.add_argument('--epochs_gcn', type=int, default=400, help='Number of training epochs per iteration.')
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
    with open('./log/param_{}.txt'.format(name), 'a') as f:
        f.write('{:}\n'.format(trial.value))
        for key, value in trial.params.items():
            f.write('{:},{:}  \n'.format(key,value)) 
     
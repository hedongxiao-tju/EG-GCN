import torch
from sklearn.metrics import normalized_mutual_info_score,f1_score
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, f1_score
def accuracy(output, labels):
    """
      计算分类准确率的函数。
    :param output: 模型的输出，通常是一个张量，大小为 [N, C]，其中 N 是样本数量，C 是类别数量。
    :param labels: 真实标签，通常是一个张量，大小为 [N]。
    :return: 准确率，一个标量值。
    """

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def eval_rocauc(y_pred,y_true):
    
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.int().detach().cpu().numpy()
    if y_true.shape[1] == 1:
       # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()
 
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

def edge_accuracy(output, labels):

    predicted_labels = torch.where(output > 0.5, 1, 0)
    correct_predictions = (predicted_labels == labels).sum().item()
    accuracy = correct_predictions / labels.size(0)

    return accuracy

def getf1(out, labels):
    label_max = out.max(1)[1].cpu()
    # label_max.append(out.max(1)[1])
    labelcpu = labels.data.cpu()
    macro_f1 = f1_score(labelcpu, label_max, average='macro')

    return macro_f1

import torch
import os
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork, Actor
import torch_geometric.transforms as T
from sklearn.metrics.pairwise import cosine_similarity as cos
from torch_geometric.utils import remove_self_loops ,is_undirected, add_self_loops
import numpy as np
import scipy.sparse as sp
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
import pickle
import pandas as pd
from sklearn.preprocessing import label_binarize
# from google_drive_downloader import GoogleDriveDownloader as gdd
from os import path
import os
from load_data import load_twitch, load_fb100, load_twitch_gamer, DATAPATH
from data_utils import rand_train_test_idx, even_quantile_labels
from homophily import our_measure, edge_homophily_edge_idx
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import is_undirected, to_undirected, one_hot
from torch_geometric.data import InMemoryDataset, download_url, Data
# from ogb.nodeproppred import NodePropPredDataset

dataset_drive_url = {
    'twitch-gamer_feat': '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges': '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',  # Wiki 1.9M
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',  # Wiki 1.9M
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'  # Wiki 1.9M
}

# 加载数据集
def load_data(data):
    if (data in ["Cora", "Pubmed", "Citeseer"]):
        # 加载数据集 Cora,Citeseer,Pubmed
        dataset = Planetoid(root='./new_pyg_data/', name=data, split='geom-gcn')
    elif (data in ["texas", "wisconsin", "cornell"]):
        # 加载数据集 Texas,Wisconsin,Cornell
        dataset = WebKB(root='./new_pyg_data/', name=data, transform=T.NormalizeFeatures())
    elif (data in ["chameleon", "squirrel"]):
        # 加载数据集Chameleon，Squirrel
        dataset = WikipediaNetwork(root='./new_pyg_data/', name=data, transform=T.NormalizeFeatures(),
                                   geom_gcn_preprocess=True)
    elif (data == "Actor"):
        # 加载数据集 Actor
        dataset = Actor(root='./new_pyg_data/Actor', transform=T.NormalizeFeatures())
    elif (data in ['genius','Penn94']):
        dataset = load_nc_dataset(data)
    # elif (data in ['ogbn-arxiv', 'ogbn-products', 'pokec', 'arxiv-year', 'genius','twitch-gamer','snap-patents','penn94']):
    #     dataset = load_nc_dataset(data)
    else:
        print("No dataset!")
    return dataset

class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(dataname):
    """ Loader for NCDataset, returns NCDataset. """
    if dataname == 'Penn94':
        dataset = load_fb100_dataset(dataname)
    elif dataname == "genius":
        dataset = load_genius()
      # elif dataname == 'twitch-e':
      #         # twitch-explicit graph
      #     dataset = load_twitch_dataset(dataname)
      # elif dataname == 'ogbn-proteins':
      #     dataset = load_proteins_dataset()
      # elif dataname == 'deezer-europe':
      #     dataset = load_deezer_dataset()
      # elif dataname == 'arxiv-year':
      #      dataset = load_arxiv_year_dataset()
      # elif dataname == 'pokec':
      #     dataset = load_pokec_mat()
      # elif dataname == 'snap-patents':
      #     dataset = load_snap_patents_mat()
      # elif dataname == 'yelp-chi':
      #     dataset = load_yelpchi_dataset()
      # elif dataname in ('ogbn-arxiv', 'ogbn-products'):
      #     dataset = load_ogb_dataset(dataname)
      # elif dataname in ('Cora', 'CiteSeer', 'PubMed'):
      #     dataset = load_planetoid_dataset(dataname)
      # elif dataname in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
      #     dataset = load_geom_gcn_dataset(dataname)
      # elif dataname == "twitch-gamer":
      #     dataset = load_twitch_gamer_dataset()
      # elif dataname == "wiki":
      #     dataset = load_wiki()

    else:
        raise ValueError('Invalid dataname')

    return dataset

def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes, 'class': 2}
    dataset.label = torch.tensor(label)
    data = Data(x=dataset.graph['node_feat'], y=torch.squeeze(dataset.label),
                edge_index=dataset.graph['edge_index'], num_nodes=dataset.graph['num_nodes'],
                num_classes=dataset.graph['class'])
    print(data.is_undirected())
    data.edge_index = to_undirected(data.edge_index)
    return data

def load_genius():
    filename = 'genius'
    dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'data/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes,
                     'class': 2}
    dataset.label = label
    data = Data(x=dataset.graph['node_feat'], y=torch.squeeze(dataset.label),
                edge_index=dataset.graph['edge_index'], num_nodes=dataset.graph['num_nodes'],
                num_class=dataset.graph['class'])
    print(data.is_undirected())
    data.edge_index =to_undirected(data.edge_index)
    return data


'''
def load_twitch_dataset(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(lang)
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset
    
def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset

def load_proteins_dataset():
    ogb_dataset = NodePropPredDataset(name='ogbn-proteins')
    dataset = NCDataset('ogbn-proteins')

    def protein_orig_split(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        return {'train': torch.as_tensor(split_idx['train']),
                'valid': torch.as_tensor(split_idx['valid']),
                'test': torch.as_tensor(split_idx['test'])}

    dataset.get_idx_split = protein_orig_split
    dataset.graph, dataset.label = ogb_dataset.graph, ogb_dataset.labels

    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['edge_feat'] = torch.as_tensor(dataset.graph['edge_feat'])
    dataset.label = torch.as_tensor(dataset.label)
    return dataset


def load_ogb_dataset(name, nclass=40):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name)
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    label = even_quantile_labels(
        dataset.graph['node_year'].flatten(), nclass, verbose=False)
    dataset.graph['label'] = torch.as_tensor(label).reshape(-1, 1)
    dataset.graph['class'] = nclass
    # Data(x=data['node_feat'], edge_index=data['edge_index'], y=torch.squeeze(data['label']),
    #      num_nodes=data['num_nodes'], num_class=data['class'])
    data = Data(x=dataset.graph['node_feat'], y=torch.squeeze(dataset.graph['label']),
                edge_index=dataset.graph['edge_index'], num_nodes=dataset.graph['num_nodes'],
                num_class=dataset.graph['class'])
    return data
    # return dataset


def load_pokec_mat():
    """ requires pokec.mat
    """
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['pokec'], \
            dest_path=f'{DATAPATH}pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes,'class': 2}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    data = Data(x=dataset.graph['node_feat'], y=torch.squeeze(dataset.label),
                edge_index=dataset.graph['edge_index'], num_nodes=dataset.graph['num_nodes'],
                num_class=dataset.graph['class'])
    print(data.is_undirected())
    data.edge_index =to_undirected(data.edge_index)
    return data


def load_snap_patents_mat(nclass=5):
    if not path.exists(f'{DATAPATH}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['snap-patents'], \
            dest_path=f'{DATAPATH}snap_patents.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{DATAPATH}snap_patents.mat')

    dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes,'class': 5}

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    dataset.label = torch.tensor(label, dtype=torch.long)
    data = Data(x=dataset.graph['node_feat'], y=torch.squeeze(dataset.label),
                edge_index=dataset.graph['edge_index'], num_nodes=dataset.graph['num_nodes'],
                num_class=dataset.graph['class'])
    print(data.is_undirected())
    data.edge_index =to_undirected(data.edge_index)
    return data


def load_yelpchi_dataset():
    if not path.exists(f'{DATAPATH}YelpChi.mat'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['yelp-chi'], \
            dest_path=f'{DATAPATH}YelpChi.mat', showsize=True)
    fulldata = scipy.io.loadmat(f'{DATAPATH}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


def load_planetoid_dataset(name):
    torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid',
                              name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    def planetoid_orig_split(**kwargs):
        return {'train': torch.as_tensor(dataset.train_idx),
                'valid': torch.as_tensor(dataset.valid_idx),
                'test': torch.as_tensor(dataset.test_idx)}

    dataset.get_idx_split = planetoid_orig_split
    dataset.label = label

    return dataset


def load_geom_gcn_dataset(name):
    fulldata = scipy.io.loadmat(f'{DATAPATH}/{name}.mat')
    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset





def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['twitch-gamer_feat'],
            dest_path=f'{DATAPATH}twitch-gamer_feat.csv', showsize=True)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['twitch-gamer_edges'],
            dest_path=f'{DATAPATH}twitch-gamer_edges.csv', showsize=True)

    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes
                     ,'class': 2}
    dataset.label = torch.tensor(label)
    data = Data(x=dataset.graph['node_feat'], y=torch.squeeze(dataset.label),
                edge_index=dataset.graph['edge_index'], num_nodes=dataset.graph['num_nodes'],
                num_class=dataset.graph['class'])

    print(data.is_undirected())
    data.edge_index =to_undirected(data.edge_index)
    print("undirected:{}".format(data.is_undirected()))
    return data


def load_wiki():
    if not path.exists(f'{DATAPATH}wiki_features2M.pt'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['wiki_features'], \
            dest_path=f'{DATAPATH}wiki_features2M.pt', showsize=True)

    if not path.exists(f'{DATAPATH}wiki_edges2M.pt'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['wiki_edges'], \
            dest_path=f'{DATAPATH}wiki_edges2M.pt', showsize=True)

    if not path.exists(f'{DATAPATH}wiki_views2M.pt'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['wiki_views'], \
            dest_path=f'{DATAPATH}wiki_views2M.pt', showsize=True)

    dataset = NCDataset("wiki")
    features = torch.load(f'{DATAPATH}wiki_features2M.pt')
    edges = torch.load(f'{DATAPATH}wiki_edges2M.pt').T
    row, col = edges
    print(f"edges shape: {edges.shape}")
    label = torch.load(f'{DATAPATH}wiki_views2M.pt')
    num_nodes = label.shape[0]

    print(f"features shape: {features.shape[0]}")
    print(f"Label shape: {label.shape[0]}")
    dataset.graph = {"edge_index": edges,
                     "edge_feat": None,
                     "node_feat": features,
                     "num_nodes": num_nodes}
    dataset.label = label

    return dataset
'''
"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
import copy

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

class GCDataset(Dataset):
    def __init__(self, dat, KP = False, normlize = False, device = 'cpu'):
        self.adjs, self.feats, self.label = dat
        self.KP = KP
        self.normlize = normlize
        self.device = device

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if type(idx) == int:
            return self.adjs[idx].to(self.device), self.feats[idx].to(self.device), self.label[idx].to(self.device), torch.LongTensor([self.adjs[idx].shape[0]]).to(self.device)
        else:
            if type(idx) == slice:
                adjs = self.adjs[idx]
                feats = self.feats[idx]
            else:
                adjs = [self.adjs[i] for i in idx]
                feats = [self.feats[i] for i in idx]
            if not self.KP:
                dic = {}
                st_idx = [0]
                for adj in adjs:
                    adj_list = adj.tolil().rows
                    for i, li in enumerate(adj_list):
                        dic[i + st_idx[-1]] = (torch.tensor(li) + st_idx[-1]).tolist()
                    st_idx.append(st_idx[-1] + adj.shape[0])
                adj = nx.to_scipy_sparse_matrix(nx.from_dict_of_lists(dic))
                ed_idx = torch.LongTensor(st_idx[1:])
                if self.normlize:
                    adj = normalize(adj + sp.eye(adj.shape[0]))
                adj = sparse_mx_to_torch_sparse_tensor(adj)
                return adj.to(self.device), torch.concat(feats, 0).to(self.device), self.label[idx].to(self.device), ed_idx.to(self.device)
            else:
                st_idx = [0]
                nei = []
                nei_mask = []
                num_nei = 0
                for adj in adjs:
                    num_nei = max(num_nei, adj.sum(1).max())
                for adj in adjs:
                    adj_list = adj.tolil().rows
                    for i, li in enumerate(adj_list):
                        nei.append(torch.concat([(torch.tensor(li) + st_idx[-1]).type(torch.int64), (torch.tensor([0] * int(num_nei - len(li)))).type(torch.int64)]))
                        nei_mask.append(torch.tensor([1] * len(li) + [0] * int(num_nei - len(li))))
                    st_idx.append(st_idx[-1] + adj.shape[0])
                ed_idx = torch.LongTensor(st_idx[1:])
                nei = torch.LongTensor(torch.stack(nei))
                nei_mask = torch.LongTensor(torch.stack(nei_mask))
                return nei.to(self.device), nei_mask.to(self.device), torch.concat(feats, 0).to(self.device), self.label[idx].to(self.device), ed_idx.to(self.device)
                    

def get_nei(adj, max_nei_ratio=None, min_nei=None):
    """
    处理稀疏邻接矩阵以生成邻居索引张量和对应的掩码张量，支持动态邻居数采样。

    Parameters:
    adj (scipy.sparse matrix): 表示图的稀疏邻接矩阵
    max_nei_ratio (float, optional): 最大邻居数与节点数的比例，用于大规模图
    min_nei (int, optional): 最小保留邻居数

    Returns:
    nei (torch.Tensor): 形状为 (n, num_nei) 的张量，每行包含对应节点的邻居索引
    nei_mask (torch.Tensor): 与nei形状相同的掩码张量，值为1表示有效邻居，0表示填充
    """
    n, _ = adj.shape
    
    # 计算原始最大邻居数
    orig_num_nei = adj.sum(1).max()
    
    # 如果设置了最大邻居比例，动态调整邻居数
    if max_nei_ratio is not None and min_nei is not None:
        # 根据节点数计算最大邻居数
        dynamic_max_nei = max(min_nei, int(n * max_nei_ratio))
        # 使用较小的值作为实际最大邻居数
        num_nei = min(orig_num_nei, dynamic_max_nei)
    else:
        num_nei = orig_num_nei
    
    adj_list = adj.tolil().rows  # 获取邻接列表
    nei = []
    nei_mask = []
    
    for li in adj_list:
        # 如果实际邻居数超过限制，进行随机采样
        if len(li) > num_nei:
            # 随机采样邻居
            sampled_indices = np.random.choice(len(li), num_nei, replace=False)
            sampled_neighbors = [li[i] for i in sampled_indices]
            nei.append(torch.tensor(sampled_neighbors))
            nei_mask.append(torch.tensor([1] * num_nei))
        else:
            # 原有邻居数未超过限制，保持不变并填充
            nei.append(torch.tensor(li + [0] * int(num_nei - len(li))))
            nei_mask.append(torch.tensor([1] * len(li) + [0] * int(num_nei - len(li))))
    
    nei = torch.stack(nei)
    nei_mask = torch.stack(nei_mask)
    
    # 打印统计信息
    if max_nei_ratio is not None:
        print(f"[INFO] 邻居采样: 原始最大邻居数 {orig_num_nei} -> 采样后 {num_nei} (节点数: {n})")
    
    return nei, nei_mask

# 新增获取邻居的函数，同时支持原始版本和优化版本
def get_nei_adaptive(adj, use_sampling=True, max_nei_ratio=0.15, min_nei=10):
    """
    自适应版本的get_nei函数，根据图规模决定是否使用邻居采样
    
    Parameters:
    adj (scipy.sparse matrix): 表示图的稀疏邻接矩阵
    use_sampling (bool): 是否使用邻居采样
    max_nei_ratio (float): 最大邻居数与节点数的比例
    min_nei (int): 最小保留邻居数
    
    Returns:
    nei, nei_mask: 邻居索引张量和掩码
    """
    n, _ = adj.shape
    
    # 对于大规模图或明确设置使用采样时应用优化
    if use_sampling and (n > 1000 or adj.sum(1).max() > 100):
        return get_nei(adj, max_nei_ratio, min_nei)
    else:
        return get_nei(adj)

def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    elif args.task == 'gc':
        data = load_data_gc(args, args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    if args.task != 'gc':
        data['adj_train_norm'], data['features'] = process(
                data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
        )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset == 'wordnet_mammal':
        adj, features = load_data_wordnet_mammal(dataset, data_path)
    elif dataset == 'wordnet_noun':
        adj, features = load_data_wordnet_noun(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        adj, features, labels = load_new_data(dataset, use_feats, data_path)
        val_prop, test_prop = 0.15, 0.15
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)
    elif dataset in ['cora', 'pubmed', 'citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.10
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################

def load_new_data(dataset_str, use_feats, data_path, split_seed=None):
    def preprocess_features(features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        rowsum = (rowsum==0)*1+rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features
    graph_adjacency_list_file_path = os.path.join(data_path, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join(data_path, 'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_str == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])

    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                            label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                            label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))
    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)
    return sp.csr_matrix(adj), features, labels


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # print(x.shape, allx.shape, tx.shape, ally.shape, ty.shape, len(graph))
    if dataset_str == 'citeseer':
        for i in range(allx.shape[0] + tx.shape[0], len(graph)):
            graph.pop(i)
        for i in range(len(graph)):
            graph[i] = [x for x in graph[i] if x < allx.shape[0] + tx.shape[0]]
    # print(len(graph))
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_reorder = [x for x in test_idx_reorder if x < allx.shape[0] + tx.shape[0]]
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + min(1000, len(labels) - len(y) - len(idx_test)))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

def load_data_wordnet_noun(dataset_str, data_path, return_label=False):
    with open(os.path.join(data_path, dataset_str + '_trans.txt')) as f:
        lines = f.readlines()
    G = nx.Graph() 
    G.add_nodes_from(range(len(lines)))
    labels = [0 for _ in range(len(lines))]
    for line in lines:
        line = line.split()
        labels[int(line[0])] = int(line[1])
        for i in range(2, len(line)):
            G.add_edge(int(line[0]), int(line[i]))
    adj = nx.adjacency_matrix(G)
    features = sp.eye(adj.shape[0])
    if return_label:
        return adj, features, np.array(labels)
    else:
        return adj, features
        
def load_data_wordnet_mammal(dataset_str, data_path):
    with open(os.path.join(data_path, dataset_str + '_trans.txt')) as f:
        lines = f.readlines()
    G = nx.Graph()
    G.add_nodes_from(range(len(lines)))
    for line in lines:
        line = line.split()
        for i in range(1, len(line)):
            G.add_edge(int(line[0]), int(line[i]))
    adj = nx.adjacency_matrix(G)
    features = sp.eye(adj.shape[0])
    return adj, features

def load_data_gc(args, dataset, use_feats, data_path, split_seed):
    if dataset in ['COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'MUTAG', 'NCI1', 'PROTEINS', 'PTC', 'REDDITBINARY', 'REDDITMULTI5K', 'DD', 'ENZYMES', 'NCI109']:
        adj, features, labels = load_gc_data(dataset, use_feats, data_path)
        # hgnn_adj, hgnn_weight = convert_hgnn_adj_gc(adj)
        val_prop, test_prop = args.val_prop, args.test_prop
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

def load_gc_data(dataset, use_feats, data_path):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    adjs = []
    feats = []
    labels = []

    with open(os.path.join(data_path, dataset + '.txt'), 'r') as f:
        n_g = int(f.readline().strip()) #first line in the dataset.txt file, the number of graphs
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]  # num of nodes and label
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped # creating label dictionary, if no match, then add new label to dict
            g = nx.Graph() # use nx from networkx to init an empty graph
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j) # create an empty node
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2 
                if tmp == len(row): #you can see from ENZYMES dataset, where this corresponds to no node attribute, only node tags
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]]) #This is a different dataset case
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    #print(j,row[k]) #This logic is experimented, correct
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))
    
    #print(g_list[0].label,g_list[1].label)#labe is correct
    #print(label_dict)

    #add labels and edge_mat       
    for g in g_list:
        #print(g.g.edges()) this is checked, up to here is correct
        #Note g is a S2VGraph element
        g.neighbors = [[] for i in range(len(g.g))] #len(g.g) returns the number of nodes
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)
        #print(g.neighbors) Correctly extract neighbors

        g.label = label_dict[g.label] #simply follow the dictionary to assign new label (but it's correct and not important)

        #print(g.g.edges())
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        #print(edges) #This correctly logs the link between nodes within every graph
        g.edge_mat = torch.LongTensor(edges).transpose(0,1) #torch geometric edge_mat format (2,2*edges), seems to be correct

    # if degree_as_tag:
    #     for g in g_list:
    #         g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    for g in g_list:
        #print("Edges in the graph:", g.g.edges())#THIS IS CAUSING PROBLEM !!!!!!! DAMN IT !!!!!!
        #adj_matrix = nx.adjacency_matrix(g.g)
        #print("Adjacency matrix (dense format):")
        #print(adj_matrix.todense())

        num_nodes = len(g.g.nodes())

        # init a zero-adjacency matrix
        adj_manual = np.zeros((num_nodes, num_nodes), dtype=int)

        # manual adjacency matrix filling
        for u, v in g.g.edges():
            adj_manual[u, v] = 1
            adj_manual[v, u] = 1  # both direcitons

        # make it a sparse matrix and save it in the list
        adj_sparse_manual = sp.csr_matrix(adj_manual)
        adjs.append(adj_sparse_manual)

        #adjs.append(sp.csr_matrix(nx.adjacency_matrix(g.g))) #It works with shape but value is wrong
        if use_feats:
            feats.append(g.node_features)
        else:
            feats.append(sp.eye(adjs[-1].shape[0]))
        labels.append(g.label)
        # tmp = np.zeros(len(label_dict))
        # tmp[g.label] = 1
        # labels.append(tmp)

    return adjs, feats, np.array(labels)


def split_batch(batch_size, data, label, nei, nei_mask):
    n, _ = nei.shape
    nei_list = []
    nei_mask_list = []
    data_list = []
    label_list = []
    for i in range(0, n, batch_size):
        ed = (i + batch_size) if i + batch_size <= n else n
        tmp = nei[i : ed, :]
        tmp_mask = nei_mask[i : ed, :]
        mask = ((tmp < ed) & (tmp >= i)).type(torch.int)
        tmp *= mask
        tmp -= mask * i
        tmp_mask *= mask
        nei_list.append(tmp)
        nei_mask_list.append(tmp_mask)
        data_list.append(data[i : ed])
        label_list.append(label[i : ed])
    return data_list, label_list, nei_list, nei_mask_list
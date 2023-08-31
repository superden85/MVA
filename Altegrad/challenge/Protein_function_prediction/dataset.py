import csv
import os
import random

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle
import bz2
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected
from tqdm import tqdm

import utils


class ProteinDataset:
    data_dir = './dataset/'

    cache_dataset = True
    cache_file = data_dir + 'dataset.pkl.bz2'

    NUM_CLASSES = 18
    NUM_NODE_FEATURES = 83  # =86-3 (removed the 3D coordinates from the features)
    NUM_EDGE_FEATURES = 5

    NUM_ACIDS = 21
    amino_acids = 'ACDEFGHIKLMNPQRSTVWXY'
    acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    MAX_SEQ_LEN = 989 + 100

    def __init__(self, batch_size, num_validation_samples=0, pretrained_seq_encoder=None, transforms=None, pca_dim=-1):
        self.batch_size = batch_size
        self.validation_samples = num_validation_samples

        if self.cache_dataset and os.path.exists(self.cache_file):
            print('Loading dataset from cache...')
            with bz2.BZ2File(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                self.protein_names = data['protein_names']
                self.protein_labels = data['protein_labels']
                self.has_label = data['has_label']
                self.sequences = data['sequences']
                self.graphs = data['graphs']
        else:
            print('Loading dataset from raw files. This may take a few minutes.')
            # Read labels
            print('\t[1/6] Reading labels')
            self.protein_names = []
            self.protein_labels = []
            self.has_label = []
            with open(self.data_dir + 'graph_labels.txt', 'r') as f:
                for i, line in enumerate(f):
                    t = line.split(',')
                    self.protein_names.append(t[0])
                    has_label = (len(t[1][:-1]) > 0)
                    self.has_label.append(has_label)
                    if has_label:
                        self.protein_labels.append(int(t[1][:-1]))
                    else:
                        self.protein_labels.append(-1)
            self.protein_names = np.array(self.protein_names)
            self.protein_labels = np.array(self.protein_labels)
            self.has_label = np.array(self.has_label)

            # Read sequences
            print('\t[2/6] Reading sequences')
            self.sequences = []
            with open(self.data_dir + 'sequences.txt', 'r') as f:
                for line in f:
                    acids = np.array([self.acid_to_index[aa] for aa in line[:-1]])
                    self.sequences.append(acids)

            # Read node attributes
            print('\t[3/6] Reading node attributes')
            node_attr = np.loadtxt(self.data_dir + "node_attributes.txt", delimiter=",")
            total_num_nodes = node_attr.shape[0]

            # Read edge attributes
            print('\t[4/6] Reading edge attributes')
            edge_attr = np.loadtxt(self.data_dir + "edge_attributes.txt", delimiter=",")

            # Read adjacency matrix
            print('\t[5/6] Reading edge list')
            edges = np.loadtxt(self.data_dir + "edgelist.txt", dtype=np.int64, delimiter=",")

            # Separate each protein graph
            print('\t[6/6] Separating graphs')
            self.graphs = []
            idx_n = 0
            idx_m = 0
            graph_indicator = np.loadtxt(self.data_dir + "graph_indicator.txt", dtype=np.int64)
            _, graph_sizes = np.unique(graph_indicator, return_counts=True)
            for i in tqdm(range(len(graph_sizes)), desc='Separating graphs', leave=False):
                n = graph_sizes[i]
                m = np.sum(np.logical_and(idx_n <= edges[:, 0], edges[:, 0] < idx_n + n))
                x_features = torch.from_numpy(node_attr[idx_n:idx_n + n]).float()
                graph_edge_index = torch.from_numpy(edges[idx_m:idx_m + m]).t().contiguous() - idx_n
                graph_edge_attr = torch.from_numpy(edge_attr[idx_m:idx_m + m]).float()

                graph_edge_index, graph_edge_attr = to_undirected(graph_edge_index, graph_edge_attr, num_nodes=n, reduce='mean')
                graph = Data(
                    x=x_features[:, 3:],  # remove 3D coordinates
                    edge_index=graph_edge_index,
                    edge_attr=graph_edge_attr,
                    pos=x_features[:, :3],
                )
                graph.validate()
                self.graphs.append(graph)

                idx_n += n
                idx_m += m

            if self.cache_dataset:
                print('(Saving dataset to load faster next time)')
                data = {
                    'protein_names': self.protein_names,
                    'protein_labels': self.protein_labels,
                    'has_label': self.has_label,
                    'sequences': self.sequences,
                    'graphs': self.graphs,
                }
                with bz2.BZ2File(self.cache_file, 'wb') as f:
                    pickle.dump(data, f)

        self.sequences = [torch.from_numpy(seq).long() for seq in self.sequences]

        all_samples = list(zip(self.sequences, self.graphs, self.protein_labels))

        # Replace node features with sequence embeddings
        if pretrained_seq_encoder is not None:
            pretrained_seq_encoder.eval()
            full_loader = DataLoader(all_samples, batch_size=pretrained_seq_encoder.config.batch_size, shuffle=False, collate_fn=batch_collate_fn)

            idx = 0
            with torch.no_grad():
                for batch in tqdm(full_loader, desc='Generating sequence embeddings from pretrained model'):
                    sequences, graphs, _ = batch
                    sequences = [s.to(pretrained_seq_encoder.device) for s in sequences]
                    graphs = graphs.to(pretrained_seq_encoder.device)
                    x = pretrained_seq_encoder(sequences, graphs).detach().cpu()
                    x = [x[i, :sequences[i].shape[0]] for i in range(x.shape[0])]  # remove padding
                    for i in range(len(x)):
                        all_samples[idx][1].x = x[i]  # replace node features with sequence embeddings
                        idx += 1

        # Apply node feature transformations
        if transforms is not None:
            for i in range(len(all_samples)):
                g = transforms(all_samples[i][1])
                all_samples[i] = (all_samples[i][0], g, all_samples[i][2])

        # Reduce dimension with PCA
        if pca_dim is not None and pca_dim > 0:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=pca_dim)
            x = torch.cat([g.x for g in self.graphs], dim=0)
            x = pca.fit_transform(x)
            idx = 0
            for i in range(len(self.graphs)):
                n = self.graphs[i].x.shape[0]
                self.graphs[i].x = torch.from_numpy(x[idx:idx + n]).float()
                idx += n

        # Split into train, test and validation
        train_data = [x for i, x in enumerate(all_samples) if self.has_label[i]]
        test_data = [x for i, x in enumerate(all_samples) if not self.has_label[i]]
        self.test_protein_names = self.protein_names[~self.has_label]
        if num_validation_samples > 0:
            # random.Random(0).shuffle(train_data)
            # val_data = train_data[:num_validation_samples]
            # train_data = train_data[num_validation_samples:]

            # stratified sampling
            train_data = np.array(train_data, dtype=object)
            train_labels = train_data[:, 2]
            train_inputs = train_data[:, :2]
            train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, test_size=num_validation_samples, stratify=train_labels, random_state=0)
            train_data = list(zip(train_inputs[:, 0], train_inputs[:, 1], train_labels))
            val_data = list(zip(val_inputs[:, 0], val_inputs[:, 1], val_labels))

        # Make dataloaders
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=batch_collate_fn)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=batch_collate_fn)
        if num_validation_samples > 0:
            self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=batch_collate_fn)
        else:
            self.val_loader = None

        print('Dataset loaded:')
        print(f'\tTrain: {len(train_data)} samples')
        print(f'\tTest: {len(test_data)} samples')
        if num_validation_samples > 0:
            print(f'\tValidation: {len(val_data)} samples')


def batch_collate_fn(batch):
    sequences, graphs, labels = zip(*batch)
    graphs = Batch.from_data_list(graphs)
    labels = torch.tensor(labels, dtype=torch.long)

    return sequences, graphs, labels

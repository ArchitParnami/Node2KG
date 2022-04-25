import os
import json
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from GraphEmbed.scripts.baseUtil import get_graph_dirs, read_adjacenyList, read_splits
from GraphEmbed.scripts.read_embeddings import read_embeddings_OpenNE, read_embeddings_OpenKE
from GraphEmbed.Config import Config

class EmbeddingData(Dataset):
    def __init__(self, dataset, min_size, max_size, source, target, dim):
        self.dataset = dataset
        self.min_size = min_size
        self.max_size = max_size
        self.source = source
        self.target = target
        self.dim = dim
        self.data = self._read_dataset()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def _read_dataset(self):
        data = []
        graph_dirs = get_graph_dirs(self.dataset, self.min_size, self.max_size)
        for graph_dir in graph_dirs:
            source_file = os.path.join(graph_dir, Config.EMBEDDINGS_DIR, str(self.dim), self.source, Config.EMBEDDINGS_FILE)
            target_file = os.path.join(graph_dir, Config.EMBEDDINGS_DIR, str(self.dim), self.source, self.target + '.json')
            adj_file  = os.path.join(graph_dir, Config.ADJ_LIST_FILE)
            source_emd = read_embeddings_OpenNE(source_file)
            target_emd = read_embeddings_OpenKE(target_file)
            adj_list = read_adjacenyList(adj_file)
            data.append((source_emd, target_emd, adj_list))
        return data


def load(dataset, graph_size, split_name, source, target, dim):
    batch_size = 1
    min_size, max_size = graph_size
    dataset = EmbeddingData(dataset, min_size, max_size, source, target, dim)
    
    train_indices, val_indices, test_indices = read_splits(split_name)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=test_sampler)
    
    return train_loader, validation_loader, test_loader
                        
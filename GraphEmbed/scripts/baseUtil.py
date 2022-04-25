import argparse
import os
from GraphEmbed.Config import Config
from GraphEmbed.OpenKE.models import *
from datetime import datetime
import ast
import torch

def basic_parser(description = ''):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--min_size', type=int, default=20,
                        help='Minimum number of nodes in the graph, default: 20')
    parser.add_argument('--max_size', type=int, default=25,
                        help='Maximum number of nodes in the graph, default: 25') 
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['amazon', 'youtube', 'dblp','lj','orkut'],
                        help='dataset in use for graphs, default: amazon')
    parser.add_argument('--log', action='store_false',
                        help='redirect standard output to a file (default:True)')
    return parser

def get_graph_dirs(dataset, min_size, max_size):
    graph_root = os.path.join(Config.PROJECT_DATASETS, 
                            dataset, Config.GRAPH_ROOT_DIRNAME, 
                            Config.GRAPH_SUBDIR_FORMAT.format(min_size,max_size))
    
    graph_dirs = os.listdir(graph_root)
    graph_dirs = sorted(graph_dirs)
    graph_dirs = [os.path.join(graph_root, graph_num) for graph_num in graph_dirs if not graph_num.startswith('.')]

    return graph_dirs

def get_model(name):
    model = None
    if name == 'TransE':
        model = TransE
    elif name == 'TransD':
        model = TransD
    elif name == 'TransR':
        model = TransR
    elif name == 'TransH':
        model = TransH
    elif name == 'DistMult':
        model = DistMult
    elif name == 'ComplEx':
        model = ComplEx
    elif name == 'RESCAL':
        model = RESCAL
    elif name == 'Analogy':
        model = Analogy
    elif name == 'SimplE':
        model = SimplE
    return model


def save_args(args, filename):
    with open(filename, 'w') as wf:
        d = vars(args)
        wf.write(str(d))

def current_datetime():
   return (datetime.now()).strftime('%Y-%m-%d %H:%M:%S')

def read_args(arg_file):
    with open(arg_file, 'r') as rf:
        args = rf.readline()
    args = ast.literal_eval(args)
    return args

def read_adjacenyList(adj_file):
    '''
        read the adjaceny file and return a dictionary
    '''
    relations = {}
    with open(adj_file, 'r') as rf:
        for line in rf:
            line =line.strip('\n')
            elements = line.split(' ')
            node = int(elements[0])
            if node not in relations:
                relations[node] = []
            
            for connection in elements[1:]:
                connection = int(connection)
                relations[node].append(connection)
    return relations

def adjList_to_matrix(N, adj_list):
    '''
        Converts the adjaceny list into matrix form
            N: Number of nodes (type: int)
            adj_list: adjacency list (type:dict)
        return N x N list
    '''
    relations = torch.zeros(N,N)
    for node, connections in adj_list.items():
        for connection in connections:
            relations[node, connection] = 1
    
    return relations

def read_adj_matrix(N, adj_file):
    relations = read_adjacenyList(adj_file)
    relations = adjList_to_matrix(N, relations)
    return relations

def transformed_file(embedding_file, target_method):
    return embedding_file + '.transformed.' + target_method

def read_splits(split_name):
    all = []
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(Config.SPLIT_ROOT, split_name, split + '.txt')
        with open(split_file, 'r') as rf:
            indices = [int(line.strip('\n')) for line in rf.readlines()]
            all.append(indices)
    return (all[0], all[1], all[2]) 

def time_it(dataset, min_size, max_size, graph_id, dim, 
            method, algorithm1, algorithm2, start_time, end_time, params=''):
    
    if not os.path.exists(Config.TIMING_FILE):
        with open(Config.TIMING_FILE, 'w') as wf:
            header='TIMSTAMP,DATASET,SIZE,GRAPH,DIM,METHOD,ALGORITHM_1,ALGORITHM_2,RUN_TIME,PARAMS\n'
            wf.write(header)
    
    with open(Config.TIMING_FILE, 'a') as wf:
        timestamp = (datetime.now()).strftime("%m/%d/%Y %H:%M:%S")
        run_time = (end_time - start_time)
        row = '{},{},{}-{},{},{},{},{},{},{},{}\n'.format(timestamp,dataset,min_size,max_size,graph_id,
                                            dim, method, algorithm1, algorithm2, run_time, params)
        wf.write(row)
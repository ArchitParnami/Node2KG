import pandas as pd
import tensorflow as tf
import numpy as np
import os
import sys
import GraphEmbed.OpenNE.src.openne.__main__ as ne
from GraphEmbed.Config import Config
from GraphEmbed.scripts.baseUtil import basic_parser, get_graph_dirs, time_it
import argparse
from tqdm import tqdm
import time

def main(args):
    '''
        builds embeddings for the all the graphs in the dataset using OpenNE
    '''    
    graph_dirs = get_graph_dirs(args.dataset, args.min_size, args.max_size)
    params = 'WalkLength={};NumWalks={}'.format(args.walk_length, args.number_walks)

    for graph_dir in tqdm(graph_dirs):
        adj_file = os.path.join(graph_dir, Config.ADJ_LIST_FILE)
        embed_dir = os.path.join(graph_dir, Config.EMBEDDINGS_DIR, str(args.representation_size), args.method)
        if not os.path.exists(embed_dir):
            os.makedirs(embed_dir)
        embed_file = os.path.join(embed_dir, Config.EMBEDDINGS_FILE)
        args.input = adj_file
        args.output = embed_file
        start_time = time.process_time()
        ne.main(args)
        end_time = time.process_time()
        time_it(args.dataset, args.min_size, args.max_size, os.path.basename(graph_dir), 
                args.representation_size, 'source', args.method, '', start_time, end_time, params=params)


def parse_args():
    # New Args
    parser = basic_parser('Builds embeddings from graphs using OpenNE')
    
    # OpenNE args
    parser.add_argument('--input', default='',
                        help='Input graph file')
    parser.add_argument('--output',
                        help='Output representation file')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=32, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--method', required=True, choices=[
        'node2vec',
        'deepWalk',
        'line',
        'gcn',
        'grarep',
        'tadw',
        'lle',
        'hope',
        'lap',
        'gf',
        'sdne'
    ], help='The learning method')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    parser.add_argument('--feature-file', default='',
                        help='The file of node features')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='the negative ratio of LINE')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--no-auto-save', action='store_true',
                        help='no save the best embeddings when training LINE')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--hidden', default=16, type=int,
                        help='Number of units in hidden layer 1')
    parser.add_argument('--kstep', default=4, type=int,
                        help='Use k-step transition probability matrix')
    parser.add_argument('--lamb', default=0.2, type=float,
                        help='lambda is a hyperparameter in TADW')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=1e-6, type=float,
                        help='alhpa is a hyperparameter in SDNE')
    parser.add_argument('--beta', default=5., type=float,
                        help='beta is a hyperparameter in SDNE')
    parser.add_argument('--nu1', default=1e-5, type=float,
                        help='nu1 is a hyperparameter in SDNE')
    parser.add_argument('--nu2', default=1e-4, type=float,
                        help='nu2 is a hyperparameter in SDNE')
    parser.add_argument('--bs', default=200, type=int,
                        help='batch size of SDNE')
    parser.add_argument('--encoder-list', default='[1000, 128]', type=str,
                        help='a list of numbers of the neuron at each encoder layer, the last number is the '
                             'dimension of the output node representation')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    if args.log:
        sys.stdout = open(Config.LOG_FILE, 'a')
    main(args)
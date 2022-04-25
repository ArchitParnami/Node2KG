import argparse
import random
import os
import sys
from GraphEmbed.scripts.datafun.CommunityGraph import CommunityGraph
from GraphEmbed.scripts.datafun.Util import *
import shutil
from GraphEmbed.Config import Config
from GraphEmbed.scripts.baseUtil import basic_parser
from tqdm import tqdm

def build(graph, min_size, max_size, num_data, test_split, val_split):
    graph_dir = os.path.join(graph.dataRoot, Config.GRAPH_ROOT_DIRNAME)
    
    if not os.path.exists(graph_dir):
        print("Creating directory: ", graph_dir)
        os.makedirs(graph_dir)
    
    graph_dir = os.path.join(graph_dir, Config.GRAPH_SUBDIR_FORMAT.format(min_size, max_size))
    
    if os.path.exists(graph_dir):
        print("Removing existing directory: ", graph_dir)
        shutil.rmtree(graph_dir)
    
    print("Creating directory: ", graph_dir)
    os.makedirs(graph_dir)

    adj_list = graph.read_graph()
    communities = graph.read_communities(min_size, max_size)
    
    if num_data != -1:
        communities = communities[:num_data]

    print("Number of Graphs: ", len(communities))

    for i, community in enumerate(tqdm(communities)):
        comm_dir = os.path.join(graph_dir, str(i))
        if not os.path.exists(comm_dir):
            os.makedirs(comm_dir)
        comm_adj_list = community_adj_list(community, adj_list)
        nodes, ids, comm_adj_list = map_to_ids(community, comm_adj_list)
        edges = adjList_to_edgeList(comm_adj_list)
        # method 3 - set val-split to 0
        train_edges, val_edges, test_edges = split(edges, test_split, val_split)
        # method 2
        # train_adj_list = edgeList_to_adjList(train_edges + val_edges)
        # adjust_train_adj_list(train_adj_list, train_edges + val_edges, ids)
        # method 1 & 3
        train_adj_list = edgeList_to_adjList(train_edges)
        adjust_train_adj_list(train_adj_list, train_edges, ids)
        write_entities(nodes, ids, os.path.join(comm_dir, 'entity2id.txt'))
        write_relations(os.path.join(comm_dir, 'relation2id.txt'))
        write_graph(comm_adj_list, os.path.join(comm_dir,'adjList.txt'))
        write_graph(train_adj_list, os.path.join(comm_dir, Config.ADJ_LIST_FILE))
        write_triples(train_edges, os.path.join(comm_dir,'train2id.txt'))
        write_triples(val_edges, os.path.join(comm_dir,'valid2id.txt'))
        write_triples(test_edges, os.path.join(comm_dir,'test2id.txt'))
        write_typeconstraints(comm_dir)
        save_plot_with_edges(train_edges, val_edges, test_edges,  os.path.join(comm_dir,'graph.png'))

def parse_args():
    parser = basic_parser('Builds graph dataset')
    parser.add_argument('--test_split', type=int, default=20,
                        help='testing split percentage, default: 20')
    parser.add_argument('--val_split', type=int, default=0,
                        help='validation split percentage, default: 0')
    parser.add_argument('--num_graphs', type=int, default=-1,
                        help='Limit the number of graphs to create, default:-1 (No limit)')

    args = vars(parser.parse_args())
    return args

def main(args):
    min_size = args['min_size']
    max_size = args['max_size']
    test_split = args['test_split']
    val_split = args['val_split']
    num_data = args['num_graphs']
    dataset = args['dataset']
    graph = CommunityGraph(dataset)
    build(graph, min_size, max_size, num_data, test_split, val_split)
    

if __name__ == '__main__':
    args = parse_args()
    if args['log']:
        sys.stdout = open(Config.LOG_FILE, 'a')
    main(args)

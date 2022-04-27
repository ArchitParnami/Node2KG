import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from GraphEmbed.Config import Config
from GraphEmbed.scripts.datafun import typeconstrain

def community_adj_list(community_nodes, graph_adjacency_list):
    '''
        Input: 
            community_nodes : List of nodes in the community
            graph_adjacency_list: Adjacency dictionary of the graph
        Output:
            Adjaceny dictionary for the community
    '''
    adj_list = {}
    for node in community_nodes:
        if node in graph_adjacency_list:
            node_adj_list = []
            for graph_node in graph_adjacency_list[node]:
                if graph_node in community_nodes:
                    node_adj_list.append(graph_node)
            if len(node_adj_list) > 0:
                adj_list[node] = node_adj_list

    return adj_list

def map_to_ids(community, adj_list):
    '''
        Create ids from 0 to number of nodes in the community
        Return adj list in terms of ids
    '''
    id = 0
    nadjlist = {}
    nodemap = {}
    old_nodes = []
    new_nodes = []
    for node in community:
        nodemap[node] = id
        old_nodes.append(node)
        new_nodes.append(id)
        id += 1
    for node, connections in adj_list.items():
        e1 = nodemap[node]
        nadjlist[e1] = []
        for connection in connections:
            e2 = nodemap[connection]
            nadjlist[e1].append(e2)
    
    return (old_nodes, new_nodes, nadjlist)

def adjList_to_strList(adj_list):
    '''
        return adjaceny list as a list of strings
    '''
    strList = []
    for node, connections in adj_list.items():
        connections = list(map(lambda x : str(x), connections))
        out = ' '.join([str(node)] + connections)
        strList.append(out)
    return strList

def adjList_to_edgeList(adj_list):
    '''
        Input: adj_list of type dictionary
        Output: List of tuples of (node, connection)
    '''
    edgeList = []
    for node, connections in adj_list.items():
        for connection in connections:
            edgeList.append((node, connection))
    return edgeList

def edgeList_to_adjList(edgeList):
    '''
        Input: List of tuples of (node, connection)
        Output: Dictionary of type {node,[connections]}
    '''
    adjList = {}
    for e1, e2 in edgeList:
        if e1 not in adjList:
            adjList[e1] = []
        adjList[e1].append(e2)
    return adjList

def write_graph(adj_list,filename):
    '''
        saves the adjacency list to a file
    '''
    strList = adjList_to_strList(adj_list)
    graph = '\n'.join(strList)
    with open(filename, 'w') as wf:
        wf.write(graph)

def make_plot(adj_list, savefile=None):
    '''
        Creates a graph from given adj list
    '''
    strList = adjList_to_strList(adj_list)
    gr = nx.parse_adjlist(strList, nodetype=int)
    fig = plt.figure()
    nx.draw(gr, with_labels=True)
    if savefile is not None:
        fig.savefig(savefile)
    else:
        plt.show()

def split(list_data, test_ratio, val_ratio, shuffle=True):
    '''
        Separates the list into train, validation and test lists
    '''
    random.seed(Config.RANDOM_SEED)
    if shuffle:
        random.shuffle(list_data)
    total = len(list_data)
    test_split = int((test_ratio * total) / 100)
    test_data = random.sample(list_data, test_split)
    train_val_data = [edge for edge in list_data if edge not in test_data]
    val_split = int(val_ratio * (total-test_split) / 100)
    val_data = random.sample(train_val_data, val_split)
    train_data = [edge for edge in train_val_data if edge not in val_data]
    return train_data, val_data, test_data

def write_triples(edges, filename, dummy=True):
    '''
        write edges(n1, n2) to a file in a form n1 n2 r
        where r is relation with default value 0
        n1 and n2 are nodes of the edge 
    '''
    n = len(edges)
    with open(filename, 'w') as wf:
        wf.write(str(n) + '\n')
        for e1, e2 in edges:
            out = '{} {} 0\n'.format(e1, e2)
            wf.write(out)


def save_plot_with_edges(train_edges, val_edges, test_edges, savefile):
    '''
        Creates a graph representing edges in different colors
        train edges are red
        validation edges are blue
        test edges are green
    '''
    G = nx.Graph()
    G.add_edges_from(train_edges, color='r')
    G.add_edges_from(val_edges, color='b')
    G.add_edges_from(test_edges, color='g')
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    fig = plt.figure()
    nx.draw(G, edge_color=colors, with_labels=True)
    fig.savefig(savefile)
    plt.close()

def adjust_train_adj_list(train_adj_list, train_edges, community):
    for node in community:
        absent_node = True
        if node in train_adj_list:
            absent_node = False
        else:
            for _, node_list in train_adj_list.items():
                if node in node_list:
                    absent_node = False
                    break
        if absent_node:
            train_adj_list[node] = []


def write_entities(nodes, ids, filename):
    '''
        Creates entity2id.txt for OpenKE
    '''
    with open(filename, 'w') as wf:
        count = "{}\n".format(len(nodes))
        wf.write(count)
        for node, id in zip(nodes, ids):
            out = "{}\t{}\n".format(node, id)
            wf.write(out)

def write_relations(filename):
    '''
        Creates relation2id.txt for OpenKE
        (default is 1 relation for amazon dataset)
    '''
    with open(filename, 'w') as wf:
        wf.write('1\n')
        wf.write('purchasedtogether\t0')

def write_typeconstraints(path):
    '''
        Creates type_constrain.txt for OpenKE
    '''

    typeconstrain.main(path)
        
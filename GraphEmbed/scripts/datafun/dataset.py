# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random
from GraphEmbed.scripts.datafun.Util import  community_adj_list, make_plot
from GraphEmbed.scripts.datafun.CommunityGraph import CommunityGraph
import numpy as np


class Dataset:
    def __init__(self, name, min_size, max_size, bin_size=5):
        self.name=name
        self.min_size = min_size
        self.max_size = max_size
        self.bin_size = bin_size
        self.graph = CommunityGraph(name) 
        self.adj_list = None
        self.communities = self.graph.read_communities(min_size, max_size)
        self.xrange =  range(self.min_size, self.max_size + 1, self.bin_size)
    
    
    def init_adj(self):
        if self.adj_list == None:
            self.adj_list = self.graph.read_graph()
    
    def community_size(self):
        hist = {}
        for comm in self.communities:
            comm_size = len(comm)
            if comm_size not in hist:
                hist[comm_size] = 1
            else:
                hist[comm_size] += 1                
        hist = sorted(hist.items())    
        return hist
    
    def community_density(self, nodes, adj_list):
        num_nodes = len(nodes)
        num_edges = 0
        for _, connection in adj_list.items():
            num_edges += len(connection)
        density = 2.0 * num_edges / (num_nodes * (num_nodes-1))
        return density
    
    def community_degree(self, nodes, adj_list):
        num_nodes = len(nodes)
        num_edges = 0
        for _, connection in adj_list.items():
            num_edges += len(connection)
        avg_degree = 2.0 * num_edges / num_nodes
        return avg_degree
    
    def analyze_size(self):
        hist = self.community_size()    
        all_sizes = []
        for comm_size, count in hist:
            l = [comm_size] * count
            all_sizes.extend(l)
       
        plt.figure()
        plt.xticks(self.xrange)
        n, bins, _ = plt.hist(all_sizes, bins=self.xrange)
        plt.xlabel('Community Size')
        plt.ylabel('Frequency')
        plt.title('Community Size vs Frequency')
        plt.show()
    
        for frequency, bin in zip(n, bins):
            out = "{}-{}\t{}".format(bin, bin + self.bin_size - 1, frequency)
            print(out)
    
    def analyze_density(self):
        """
            Measures ratio of number of edges to maximum possible edges in the graph
        """
        self.init_adj()
        size_density = {}
        for community in self.communities:
            comm_adj_list = community_adj_list(community, self.adj_list)
            density = self.community_density(community, comm_adj_list)
            comm_size = len(community)
            if comm_size not in size_density:
                size_density[comm_size] = []
            size_density[comm_size].append(density)
        
        for comm_size, density_list in size_density.items():
            avg_density = sum(density_list) / len(density_list)
            size_density[comm_size] = avg_density
        
        size_density = sorted(size_density.items())
        x = [item[0] for item in size_density]
        y = [item[1] for item in size_density]
        y_avg = np.mean(y)
        plt.figure()
        plt.xticks(self.xrange)
        plt.plot(x, y)
        plt.hlines(y_avg, xmin=self.min_size, xmax=self.max_size)
        plt.xlabel('Community Size')
        plt.ylabel('Average Density')
        plt.title('Community Size vs Average Density')
        plt.show()
        print("Average density of graphs in size range {}-{} = {}".format(self.min_size, self.max_size,y_avg))

    def analyze_degree(self):
        """
            Measures ratio of number of edges to number of verticies 
        """
        self.init_adj()
        size_degree = {}
        for community in self.communities:
            comm_adj_list = community_adj_list(community, self.adj_list)
            degree = self.community_degree(community, comm_adj_list)
            comm_size = len(community)
            if comm_size not in size_degree:
                size_degree[comm_size] = []
            size_degree[comm_size].append(degree)
        
        for comm_size, degree_list in size_degree.items():
            avg_degree = sum(degree_list) / len(degree_list)
            size_degree[comm_size] = avg_degree
        
        size_degree = sorted(size_degree.items())
        x = [item[0] for item in size_degree]
        y = [item[1] for item in size_degree]
        y_avg = np.mean(y)
        plt.figure()
        plt.xticks(self.xrange)
        plt.plot(x, y)
        plt.hlines(y_avg, xmin=self.min_size, xmax=self.max_size)
        plt.xlabel('Community Size')
        plt.ylabel('Average degree')
        plt.title('Community Size vs Average degree')
        plt.show()
        print("Average degree of graphs in size range {}-{} = {}".format(self.min_size, self.max_size,y_avg))
   
    def plot_random_graph(self, min_size, max_size):
        self.init_adj()
        my_communities = [comm for comm in self.communities \
                          if len(comm) >= min_size and len(comm) <= max_size]
        community = random.sample(my_communities, 1)[0]
        comm_adj_list = community_adj_list(community, self.adj_list)
        make_plot(comm_adj_list)
    
    def size_trend(self):
        hist = self.community_size()
        X = []; Y = []
        for x, y in hist:
            X.append(x)
            Y.append(y)
        plt.figure()
        plt.plot(X, Y)
        plt.xlabel('Community Size')
        plt.ylabel('Frequency')
        plt.title('Community Size vs Frequency')
        plt.show()
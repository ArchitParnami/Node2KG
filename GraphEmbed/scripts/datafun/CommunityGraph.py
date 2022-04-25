import os
from GraphEmbed.Config import Config

class CommunityGraph:
    def __init__(self, dataset):
       self.dataRoot =  os.path.join(Config.PROJECT_DATASETS, dataset)
       self.mainGraphFile = os.path.join(self.dataRoot, Config.DATASET_GRAPH_FILE.format(dataset))
       self.commGraphFile = os.path.join(self.dataRoot, Config.DATASET_COMMUNITY_FILE.format(dataset))
    
    
    def read_graph(self):
        '''
            Read the graph file and return an adjaceny list dictionary.
            Keys are nodes in the graph.
            Values are list of connecting nodes
        '''
        adjacency_list = {}
        with open(self.mainGraphFile, 'r') as rf:
            for i in range(4):
                rf.readline()
            for line in rf:
                line = line.strip('\n')
                nodes = line.split('\t')
                e1 = nodes[0]
                e2 = nodes[1]
                if e1 not in adjacency_list:
                    adjacency_list[e1] = []
                adjacency_list[e1].append(e2)
        return adjacency_list
    
    def read_communities(self, min_size, max_size):
        '''
            Input: CommunityFile
            Output: List of Community (list of lists) 
        '''
        communities = []
        with open(self.commGraphFile, 'r') as rf:
            for line in rf:
                line = line.strip()
                items = line.split('\t')
                comm_size = len(items)
                if comm_size >= min_size and comm_size <= max_size:
                    communities.append(items)
        return communities
    

        



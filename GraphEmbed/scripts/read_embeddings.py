import torch
import json

def read_embeddings_OpenNE(embedding_file):
    embeddings = None
    with open(embedding_file, 'r') as rf:
        for i, line in enumerate(rf):
            line = line.strip('\n')
            info = line.split(' ')
            if i==0:
                n = int(info[0])
                dim = int(info[1])
                embeddings = torch.zeros(n,dim)
            else:
                index = int(info[0])
                embedding = [float(x) for x in info[1:]]
                embeddings[index] = torch.tensor(embedding)
    return embeddings

def read_embeddings_OpenKE(embedding_file):
    with open(embedding_file, 'r') as rf:
        data_dict = rf.readline()
        data_dict = json.loads(data_dict.strip('\n'))
        embeddings = list(data_dict['ent_embeddings.weight'])
        embeddings = torch.tensor(embeddings)
        return embeddings
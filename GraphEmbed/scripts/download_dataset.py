datasets={
    'amazon': [
        'http://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz', 
        'http://snap.stanford.edu/data/bigdata/communities/com-amazon.all.dedup.cmty.txt.gz'
    ],

    'youtube': [
        'http://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz',
        'http://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz'
    ],

    'dblp': [
        'http://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz',
        'http://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz'
    ],

    'orkut':[
        'http://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz',
        'http://snap.stanford.edu/data/bigdata/communities/com-orkut.all.cmty.txt.gz'
    ],

    'lj' : [
        'http://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz',
        'http://snap.stanford.edu/data/bigdata/communities/com-lj.all.cmty.txt.gz'
    ]
}


import wget
from GraphEmbed.Config import Config
import os
import gzip
import shutil
import argparse

def verify_and_download(txt_file, url):
    if not os.path.exists(txt_file):
        gz_file = txt_file + '.gz'
        if not os.path.exists(gz_file):
            print("Downloading {}".format(url))
            wget.download(url, out=gz_file)
            print()
        else:
            print('Found {}!'.format(gz_file))    
        
        print("Extracting {}".format(gz_file))
        
        with gzip.open(gz_file, 'rb') as f_in:
            with open(txt_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print('{} already exists'.format(txt_file))

def parse_args():
    parser = argparse.ArgumentParser(description='Download the specified dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['amazon', 'youtube', 'dblp','lj','orkut'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    urls = datasets[dataset]
    data_dir = os.path.join(Config.PROJECT_DATASETS, dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    graph_file = os.path.join(data_dir, Config.DATASET_GRAPH_FILE.format(dataset))
    comm_file = os.path.join(data_dir, Config.DATASET_COMMUNITY_FILE.format(dataset))
    verify_and_download(graph_file, urls[0])
    verify_and_download(comm_file, urls[1])
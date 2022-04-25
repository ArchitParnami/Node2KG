import os
import torch.optim as optim
import torch
from GraphEmbed.models.SelfAttention import SelfAttention
import pandas as pd
import argparse
from GraphEmbed.Config import Config
from GraphEmbed.scripts.baseUtil import read_args, get_graph_dirs, read_adj_matrix, transformed_file, basic_parser, time_it
from GraphEmbed.scripts.read_embeddings import read_embeddings_OpenNE
import sys
from tqdm import tqdm
import time

def load_model(model_path, dim):
    model = SelfAttention(dim)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    return model

def save_embeddings(embeddings, filename):
    with open(filename, 'w') as wf:
        n, dim = embeddings.shape
        wf.write('{} {}\n'.format(n, dim))
        for i, embedding in enumerate(embeddings):
            wf.write(str(i) + ' ')
            str_e = ' '.join([str(e) for e in embedding])
            wf.write(str_e + '\n')

def transform(graph_dir, model, embedding_source, embedding_target, dim):
    embedding_file = os.path.join(graph_dir, Config.EMBEDDINGS_DIR, str(dim), embedding_source, Config.EMBEDDINGS_FILE)
    embeddings = read_embeddings_OpenNE(embedding_file)
    adj_file = os.path.join(graph_dir, Config.ADJ_LIST_FILE)
    relations =  read_adj_matrix(embeddings.size()[0], adj_file)
    transformed_embeddings = model(embeddings, relations)
    emb_np = transformed_embeddings.detach().numpy()
    save_path = transformed_file(embedding_file, embedding_target)
    save_embeddings(emb_np, save_path)

def main(args):
    model_dir = os.path.join(Config.MODEL_SAVE, args.dataset,
        Config.GRAPH_SUBDIR_FORMAT.format(args.min_size, args.max_size),
        str(args.dim),
        args.source + '-' + args.target,
        args.model_dir)
    model_path = os.path.join(model_dir,Config.MODEL_FILE)
    opt_file = os.path.join(model_dir, Config.ARGS_FILE)
    opt = read_args(opt_file)
    model = load_model(model_path, args.dim)
    graph_dirs = get_graph_dirs(opt['dataset'], opt['min_size'], opt['max_size'])
    embedding_source = opt['source']
    embedding_target = opt['target']
    params = 'Model={}'.format(args.model_dir)
    for graph_dir in tqdm(graph_dirs):
        start_time = time.process_time()
        transform(graph_dir, model, embedding_source, embedding_target, args.dim)
        end_time = time.process_time()
        time_it(args.dataset, args.min_size, args.max_size, os.path.basename(graph_dir), args.dim,
            'transformed', embedding_source, embedding_target, start_time, end_time,params=params)

def parse_args():
    parser = basic_parser(description='Use the transformation model to generate embeddings')
    parser.add_argument('--source', default='node2vec', choices=[
                        'node2vec','deepWalk','line','gcn','grarep','tadw',
                        'lle','hope','lap','gf','sdne'], 
                        help='OpenNE method used to obtain the source embeddings. (default:node2vec)')
    parser.add_argument('--target', default='TransE', choices=[
                        'RESCAL','DistMult','Complex','Analogy','TransE',
                        'TransH','TransR','TransD','SimplE'], 
                        help='OpenKE method used to obtain the target embeddings (default:TransE)')
    parser.add_argument('--model_dir', required=True, type=str,
                        help='Name of the model directory present in models/saved/[dataset]/[graph-size]/[source-target]')
    parser.add_argument('--dim', required=True, type=int,
                        help='Embedding dimension size. (default:32)') 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.log:
        sys.stdout = open(Config.LOG_FILE, 'a')
    main(args)
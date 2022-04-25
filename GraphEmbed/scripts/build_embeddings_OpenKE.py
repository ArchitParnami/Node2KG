import os 
import argparse
import torch
import GraphEmbed.OpenKE.config as OpenKEConfig
from GraphEmbed.scripts.baseUtil import basic_parser, get_graph_dirs, get_model, time_it
from GraphEmbed.Config import Config
from tqdm import tqdm
import sys
import time

def train_model(graph_dir, args):
    con = OpenKEConfig.Config()
    con.set_in_path(graph_dir + '/')
    con.set_work_threads(8)
    con.set_train_times(args.epochs)
    con.set_nbatches(1)	
    con.set_alpha(0.001)
    con.set_bern(0)
    con.set_dimension(args.dim)
    con.set_margin(1.0)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("SGD")
    con.set_save_steps(args.epochs + 1)
    con.set_valid_steps(args.epochs + 1)
    con.set_early_stopping_patience(10)
    
    if args.use_openne:
        result_dir = os.path.join(graph_dir, Config.EMBEDDINGS_DIR, str(args.dim), args.openne_embedding)
        con.use_pretrained_embeddings(True)
        embedding_file = os.path.join(result_dir, Config.EMBEDDINGS_FILE)
        con.set_pretrain_embedding_path(embedding_file)
    else:
        result_dir = os.path.join(graph_dir, Config.EMBEDDINGS_DIR, str(args.dim))
        if not os.path.exists(result_dir):
            try:
                os.makedirs(result_dir)
            except FileExistsError:
                pass
    
    con.set_result_dir(result_dir)
    con.set_test_link(False)
    con.set_test_triple(False)
    con.use_gpu(args.cuda)
    con.init()
    con.set_train_model(get_model(args.train_method))
    con.train()
    con.unload()

def main(args):
    if args.cuda and not torch.cuda.is_available():
        print("Warning GPU Device Not Available!")
        args.cuda = False
    
    if args.use_openne:
        method="source2target"
        algorithm1 = args.openne_embedding
        algorithm2 = args.train_method
    else:
        method = "target"
        algorithm1 = args.train_method
        algorithm2 = ''
    
    params = 'Epochs={}'.format(args.epochs)
    graph_dirs = get_graph_dirs(args.dataset, args.min_size, args.max_size)
    for graph_dir in tqdm(graph_dirs):
       start_time = time.process_time()
       train_model(graph_dir, args)
       end_time = time.process_time()
       time_it(args.dataset, args.min_size, args.max_size, os.path.basename(graph_dir), 
                args.dim, method, algorithm1, algorithm2, start_time, end_time, params=params)

def parse_args():
    parser = basic_parser('Builds embeddings from graphs using OpenKE')
    parser.add_argument('--dim', default=32, type=int,
                        help='Embedding dimension size. (default:32)')
    parser.add_argument('--train_method', default='TransE', choices=[
                        'RESCAL','DistMult','Complex','Analogy','TransE',
                        'TransH','TransR','TransD','SimplE'], 
                        help='OpenKE method for training the embeddings (default:TransE)')
    parser.add_argument('--use_openne', action='store_true',
                        help='use pretrained embeddings from OpenNE. (default:False)')
    parser.add_argument('--openne_embedding', default='node2vec', choices=[
                        'node2vec','deepWalk','line','gcn','grarep','tadw',
                        'lle','hope','lap','gf','sdne'], 
                        help='OpenNE method used to obtain the embeddings')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs for training')
    parser.add_argument('--cuda', action='store_true',
                        help='use GPU for training. (default:False)')
    
    
    args = parser.parse_args()

    return args
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    args = parse_args()
    if args.log:
        sys.stdout = open(Config.LOG_FILE, 'a')
    main(args)
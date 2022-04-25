import os 
import argparse
import torch
import sys
from GraphEmbed.Config import Config
import GraphEmbed.OpenKE.config as OpenKEConfig
from GraphEmbed.scripts.baseUtil import basic_parser, get_graph_dirs, get_model, save_args, transformed_file
from tqdm import tqdm

def eval_model(graph_dir, args):
    con = OpenKEConfig.Config()
    con.set_in_path(graph_dir + '/')
    con.set_dimension(args.dim)
    embed_dir = os.path.join(graph_dir, Config.EMBEDDINGS_DIR, str(args.dim))

    if args.embedding_type in ['openne', 'transformed']:
        result_dir = os.path.join(embed_dir, args.openne_type)
        embedding_file = os.path.join(result_dir, Config.EMBEDDINGS_FILE)
        
        if args.embedding_type == 'transformed':
            embedding_file = transformed_file(embedding_file, args.eval_method)
        
        con.use_pretrained_embeddings(True)
        con.set_pretrain_embedding_path(embedding_file)
   
    elif args.embedding_type == 'openke':
        result_dir = embed_dir
    
    elif args.embedding_type == 'openne_openke':
        result_dir = os.path.join(embed_dir, args.openne_type)
    
    else:
        raise Exception('Unknown Embedding')

    con.set_result_dir(result_dir)
    con.set_test_link(True)
    con.set_test_triple(False)
    con.use_gpu(args.cuda)
    con.init()
    con.set_test_model(get_model(args.eval_method))
    con.test()
    con.unload()



def main(args):
    if args.cuda and not torch.cuda.is_available():
        print("Warning GPU Device Not Available!")
        args.cuda = False
    graph_dirs = get_graph_dirs(args.dataset, args.min_size, args.max_size)
    
    for graph_dir in tqdm(graph_dirs):
       eval_model(graph_dir, args)

def parse_args():
    parser = basic_parser('Evaluate embeddings using OpenKE')
    
    parser.add_argument('--dim', default=32, type=int,
                        help='Embedding dimension size. (default:32)')
    
    parser.add_argument('--eval_method', default='TransE', choices=[
                        'RESCAL','DistMult','Complex','Analogy','TransE',
                        'TransH','TransR','TransD','SimplE'], 
                        help='OpenKE method for evaluating the embeddings (default:TransE)')
    
    parser.add_argument('--embedding_type', required=True, choices=[
                        'openne',
                        'openke',
                        'openne_openke',
                        'transformed'], 
                        help='''Describes the source of the embeddings.
                            openne => embeddings are obtained from OpenNE.
                            openke => embeddings were purely obtained from OpenKE.
                            openne_openke => OpenNE embeddings were finetuned using OpenKE.
                            transformed => OpenNE embeddings were transformed.
                        ''')
    
    parser.add_argument('--openne_type', default='node2vec', choices=[
                        'node2vec','deepWalk','line','gcn','grarep','tadw',
                        'lle','hope','lap','gf','sdne'], 
                        help='OpenNE method used to obtain the embeddings. Required when embedding_type is [openne, openne_openke, transformed]')
    
    parser.add_argument('--cuda', action='store_true',
                        help='use GPU for training. (default:False)')
    
    parser.add_argument('--result_path', required=True,
                        help='Directory for args.txt and log.txt files')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    args = parse_args()
    arg_file = os.path.join(args.result_path, Config.ARGS_FILE)
    log_file = os.path.join(args.result_path, 'log.txt')
    sys.stdout = open(log_file, 'w')
    save_args(args, arg_file)
    main(args)
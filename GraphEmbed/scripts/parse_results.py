import argparse
import os
import GraphEmbed.Config
from GraphEmbed.scripts.baseUtil import get_graph_dirs, read_args, transformed_file
from GraphEmbed.Config import Config
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Parse results.txt file and place the results in graph folder')
    parser.add_argument('--path', required=True, type=str,
                        help='Path to directory containing results.txt')
    parser.add_argument('--log', action='store_false',
                        help='redirect standard output to a file (default:True)')
    args = parser.parse_args()
    return args


def yield_results(result_file):
    with open(result_file, 'r') as rf:
        lines = rf.readlines()
    count = 0
    for i, line in enumerate(lines):
        if line.startswith("Input Files Path"):
            path = (line.split(':')[1]).strip('\n').strip()
        if line.startswith("no type"):
            start_index = i
        if line.startswith("averaged(filter)"):
            if count == 1:
                result = lines[start_index : i+1]
                count = 0
                yield (path, result)
            else:
                count = count + 1
            
def parse_results(arg_file, result_file):
    opt = read_args(arg_file)
    
    if opt['embedding_type'] in ['openne', 'transformed']:
        embed_file = os.path.join(Config.EMBEDDINGS_DIR, str(opt['dim']), opt['openne_type'], Config.EMBEDDINGS_FILE)
        if opt['embedding_type'] == 'transformed':
            embed_file = transformed_file(embed_file, opt['eval_method'])
        else:
            embed_file += '.' + opt['eval_method']
    elif opt['embedding_type'] == 'openne_openke':
        embed_file = os.path.join(Config.EMBEDDINGS_DIR, str(opt['dim']), opt['openne_type'], opt['eval_method'] + '.json')
    else:
        embed_file = os.path.join(Config.EMBEDDINGS_DIR, str(opt['dim']), opt['eval_method'] + '.json')
    
    for path, result in yield_results(result_file):
        result_file = os.path.join(path, embed_file + '.result')
        with open(result_file, 'w') as wf:
            wf.writelines(result)

def main(args):
    result_file = os.path.join(args.path, 'results.txt')
    args_file = os.path.join(args.path, Config.ARGS_FILE)
    if not os.path.exists(result_file):
        print("File Not Found: ", result_file)
    if not os.path.exists(args_file):
        print("File Not Found: ", args_file)
    
    parse_results(args_file, result_file)

if __name__ == '__main__':
    args = parse_args()
    if args.log:
        sys.stdout = open(Config.LOG_FILE, 'a')
    main(args)
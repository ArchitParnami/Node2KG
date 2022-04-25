import numpy as np
import os
from GraphEmbed.scripts.baseUtil import basic_parser, get_graph_dirs, save_args, current_datetime
from GraphEmbed.Config import Config
from datetime import datetime
import shutil
import sys

def main(args):
    num_datasets = len(get_graph_dirs(args.dataset, args.min_size, args.max_size))
    indices = list(range(num_datasets))
    validation_split = args.val_ratio
    test_split = args.test_ratio
    shuffle_dataset = args.shuffle
    random_seed= args.seed
   
    split = int(np.floor(test_split * num_datasets))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    trainval_indices, test_indices = indices[split:], indices[:split]
    split = int(np.floor(validation_split * len(trainval_indices)))
    train_indices, val_indices = trainval_indices[split:], trainval_indices[:split]

    split_name =  current_datetime() if args.name == '' else args.name 
    split_dir = os.path.join(Config.SPLIT_ROOT, split_name)
    if os.path.exists(split_dir):
        print("Removing Current Split:", split_dir)
        shutil.rmtree(split_dir)

    os.makedirs(split_dir)

    for indices, split in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
        split_file = os.path.join(split_dir, split + '.txt')
        with open(split_file, 'w') as wf:
            line = '\n'.join([str(x) for x in indices])
            wf.write(line)
    
    save_args(args, os.path.join(split_dir, Config.ARGS_FILE))


def parse_args():
    parser = basic_parser('Generates training, validation, test splits for a given dataset')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='data split ratio for testing. (default:0.2)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='data split ratio for validation. (default:0.2)')
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED,
                        help='seed value for random')
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle the dataset before splitting. (default: False)')
    parser.add_argument('--name', type=str, default='',
                        help='Give a name to your split folder. (default: datetime.now())')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.log:
        sys.stdout = open(Config.LOG_FILE, 'a')
    main(args)
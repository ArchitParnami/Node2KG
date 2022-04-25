import torch.optim as optim
import torch
import EmbeddingDataset
from GraphEmbed.models.SelfAttention import SelfAttention
from torch.autograd import Variable
import os
from GraphEmbed.scripts.baseUtil import basic_parser, save_args, current_datetime, adjList_to_matrix, read_splits, time_it
from GraphEmbed.Config import Config
import shutil
import copy
import sys
from tqdm import tqdm
import math
import time

def euclidean_loss(x1, x2):
    return torch.mean(torch.pow(x1-x2, 2).sum(1))

def cal_loss(model, data, loss_fun):
    source_embeddings, target_embeddings, adj_list = data
    N = source_embeddings.size()[1]
    relations = adjList_to_matrix(N, adj_list)
        
    x = Variable(source_embeddings[0])
    x1 = model(x, relations)
    x2 = target_embeddings[0]
    if isinstance(loss_fun, torch.nn.CosineEmbeddingLoss):
        y = torch.ones(len(x2))
        loss = loss_fun(x1,x2,y)
    else:
        loss = loss_fun(x1,x2)
    return loss

def train(train_loader, validation_loader, model, optimizer, loss_fun, args):
    best_model_dict = None
    min_val_loss = float('inf')
    train_size = len(read_splits(args.split_name)[0])
    stop = False
    train_loss = None
    train_patience = 1
    previous_loss = 0
    i = 0

    for epoch in tqdm(range(args.epochs)):
        batch_loss = 0
        optimizer.zero_grad()
        
        for i, data in enumerate(train_loader, 0):
            loss = cal_loss(model, data, loss_fun)
            batch_loss += loss
            
            if (i+1) % args.batch_size == 0 or (i+1 == train_size):
                batch_size = args.batch_size if (i+1) % args.batch_size == 0 else (i+1) % args.batch_size
                loss = batch_loss / batch_size
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = 0
                
                total_val_loss = 0
                for k, valdata in enumerate(validation_loader, 0):
                    val_loss = cal_loss(model, valdata, loss_fun)
                    total_val_loss += val_loss
                avg_loss = total_val_loss / (k+1)
                avg_loss = avg_loss.item()
                
                print('[%d,%d] Train loss: %.3f, Val Loss: %.3f' % (epoch + 1, i + 1, loss.item(), avg_loss))


                if round(avg_loss,3) == previous_loss:
                    train_patience += 1
                else:
                    train_patience = 1
                
                previous_loss = round(avg_loss,3)
                
                
                if math.isnan(loss.item()):
                    stop=True
                    break

                if train_patience == args.patience:
                    print('Patience {} reached. Stopping the training.'.format(train_patience))
                    stop=True
                    break

                if avg_loss < min_val_loss:
                    min_val_loss = avg_loss
                    best_model_dict = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    best_batch = i + 1
                    train_loss = loss.item()
                
        if stop:
            break
    
    if best_model_dict == None:
        best_model_dict= model.state_dict()
        best_epoch = epoch + 1 
        best_batch = i + 1
    
    print('Finished Training')
    print('Minimum Loss = {:.3f} at [{},{}]'.format(min_val_loss, best_epoch, best_batch))
    
    results = {}
    results['best_model_dict'] = best_model_dict
    results['best_epoch'] = best_epoch
    results['best_batch'] = best_batch
    results['train_loss'] = train_loss
    results['val_loss'] = min_val_loss

    return results

def test(model, test_loader, loss_fun):
    print('Testing..')
    total_test_loss = 0
    k  = 0
    for k, test_data in enumerate(test_loader, 0):
        test_loss = cal_loss(model, test_data,loss_fun)
        total_test_loss += test_loss
    avg_loss = total_test_loss / (k+1)
    test_loss = avg_loss.item()
    print("Test Loss: %.3f" % (test_loss))
    return test_loss

def save_model(model_dict, optimizer, args):
    curr_dir = current_datetime() if args.name == '' else args.name
    model_dir = os.path.join(Config.MODEL_SAVE, 
                            args.dataset, 
                            Config.GRAPH_SUBDIR_FORMAT.format(args.min_size, args.max_size),
                            str(args.dim),
                            args.source + '-' + args.target,
                            curr_dir)
    if os.path.exists(model_dir):
        print("Removing the existing directory:", model_dir)
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    model_path = os.path.join(model_dir, Config.MODEL_FILE)
    arg_path = os.path.join(model_dir, Config.ARGS_FILE)
    print("Saving the model as: ", model_path)
    
    torch.save(model_dict, model_path)
    save_args(args, arg_path)
    return model_path
    
def load_model(args, model_path):
    if args.model == 'self_attention':
        model = SelfAttention(args.dim)
    model.load_state_dict(torch.load(model_path))
    return model

def parse_args():
    parser = basic_parser('Train transformation model')
    parser.add_argument('--dim', default=32, type=int,
                        help='Embedding dimension size. (default:32)')
    parser.add_argument('--epochs', type=int, default=6,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training. (default:64)')
    parser.add_argument('--model', default='self_attention', choices=['self_attention'],
                        help='Transformation model for training')
    parser.add_argument('--source', default='node2vec', choices=[
                        'node2vec','deepWalk','line','gcn','grarep','tadw',
                        'lle','hope','lap','gf','sdne'], 
                        help='OpenNE method used to obtain the source embeddings. (default:node2vec)')
    parser.add_argument('--target', default='TransE', choices=[
                        'RESCAL','DistMult','Complex','Analogy','TransE',
                        'TransH','TransR','TransD','SimplE'], 
                        help='OpenKE method used to obtain the target embeddings (default:TransE)')
    parser.add_argument('--split_name', required=True,
                        help='Name of the directory present in split/ for train, val & test information')
    parser.add_argument('--name', type=str, default='',
                        help='Give a name to your model directory. (default: datetime.now())')
    parser.add_argument('--loss', type=str, required=True, choices=['euclidean', 'mse', 'cosine'],
                        help='loss function for training the transformation')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of times to wait before stopping the training when validation loss doesnt change')
    args = parser.parse_args()
    return args

def main(args):
    train_loader, val_loader, test_loader = EmbeddingDataset.load(
        dataset=args.dataset,
        graph_size=(args.min_size, args.max_size),
        split_name = args.split_name,
        source = args.source, 
        target = args.target,
        dim=args.dim
    )
    
    if args.model == 'self_attention':
        model = SelfAttention(args.dim)
    
    if args.loss == 'euclidean':
        loss_fun = euclidean_loss
    elif args.loss == 'mse':
        loss_fun = torch.nn.MSELoss()
    elif args.loss == 'cosine':
        loss_fun = torch.nn.CosineEmbeddingLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    start_time = time.process_time()
    results = train(train_loader, val_loader, model, optimizer, loss_fun, args)
    end_time = time.process_time()
    
    model_path = save_model(results['best_model_dict'], optimizer, args)
    best_model = load_model(args, model_path)
    results.pop('best_model_dict')
    
    test_loss = test(best_model, test_loader, loss_fun)
    results['test_loss'] = test_loss
   
    time_it(args.dataset, args.min_size, args.max_size, 'TRAIN', args.dim,
            'train-transform', args.source, args.target, start_time, end_time,params=str(results))
   

if __name__ == '__main__':
    args = parse_args()
    if args.log:
        sys.stdout = open(Config.LOG_FILE, 'a')
    main(args)
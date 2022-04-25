import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from GraphEmbed.scripts.baseUtil import basic_parser, read_splits, get_graph_dirs, transformed_file
from GraphEmbed.Config import Config
from tqdm import tqdm
import sys
from GraphEmbed.Config import Config
import itertools
import matplotlib.cbook as cbook
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

measures = ["MRR", "MR", "Hit@10", "Hit@3", "Hit@1"]

keys_1 = ["no_type_constraint", "type_constraint"]
keys_2 = ["raw", "filter"]
keys_3 = ["l", "r", "average"]

def check_file_exists(filename, metrics):
    result_file = os.path.join(Config.RESULT_DIR, filename)
    if not os.path.exists(result_file):
        with open(result_file, 'w') as rf:
            header="DATASET,SIZE,DIM,SOURCE,TARGET,SPLIT,MEASURE,METHOD," + ','.join(metrics) + '\n'
            rf.write(header)

def write_row(filename, measure, method, vals, args):
    result_file = os.path.join(Config.RESULT_DIR, filename)
    row = '{},{},{},{},{},{},{},{},'.format(args.dataset, Config.GRAPH_SUBDIR_FORMAT.format(args.min_size, args.max_size),
    args.dim,args.source, args.target, args.plot_split, measure, method)
    row += ','.join(vals) + '\n'    
    with open(result_file, 'a') as rf:
        rf.write(row)

def get_result_dict():    
    result_1 = {}
    for key_1 in keys_1:
        result_2 = {}
        for key_2 in keys_2:
            result_3 = {}
            for key_3 in keys_3:
                result_3[key_3] = []
            result_2[key_2] = result_3
        result_1[key_1] = result_2
    return result_1

def get_nums(line):
    line = (line.strip('\n')).split('\t')
    items = [item.strip() for item in line if item.strip() != '']
    data = [float(num) for num in items[-5:]]
    return data

def parse_result(result_file):
    result = get_result_dict()
    with open(result_file, 'r') as rf:
        for key_1 in keys_1:
            rf.readline() # constraint type
            rf.readline() # header
            for key_2 in keys_2:
                for key_3 in keys_3:
                    r = rf.readline() # data
                    result[key_1][key_2][key_3] = get_nums(r)
                rf.readline() # blank line
    return result

def load_results(args, methods, labels):
    graph_dirs = get_graph_dirs(args.dataset, args.min_size, args.max_size)
    graph_dirs = np.array(graph_dirs)
    
    if args.plot_split != 'all':
        train_indices, val_indices, test_indices = read_splits(args.split_name)
        if args.plot_split == 'train':
            graph_dirs = graph_dirs[train_indices]
        elif args.plot_split == 'val':
           graph_dirs = graph_dirs[val_indices]
        else:
            graph_dirs = graph_dirs[test_indices]
    
    num_datasets = len(graph_dirs)
    num_measures = len(measures)
    results = np.zeros((len(methods), num_datasets, num_measures))

    for i, graph_dir in enumerate(tqdm(graph_dirs, desc="loading")):
        embed_dir = os.path.join(graph_dir, Config.EMBEDDINGS_DIR, str(args.dim), args.source)
        result_files = []
        for method in methods:
            if method == "source":
                result_file = os.path.join(embed_dir, Config.EMBEDDINGS_FILE + '.' + args.target + '.result')
            elif method == "source2target":
                result_file = os.path.join(embed_dir, args.target + '.json.result')
            elif method == "transformed":
                result_file = transformed_file(os.path.join(embed_dir, Config.EMBEDDINGS_FILE), args.target) + '.result'
            elif method == "target":
                result_file = os.path.join(graph_dir, Config.EMBEDDINGS_DIR, str(args.dim), args.target + '.json.result')
            result_files.append(result_file)

        for j, result_file in enumerate(result_files):
            results[j, i] = parse_result(result_file)["no_type_constraint"]["filter"]["average"]
    
    return results

def residual_plot(results, fig, labels, colors, methods, args):
    _, num_datasets, num_measures = results.shape
    x = range(num_datasets)

    metrics = ['mean', 'iqr', 'cilo', 'cihi', 'whishi', 'whislo', 'q1', 'med', 'q3']
    check_file_exists(Config.RESIDUAL_FILE, metrics)
    
    for i in range(num_measures):
        ax = fig.add_subplot(num_measures, 1, i+1)
        y1 = results[0,:, i]
        y2 = results[1,:, i]
        residue = y1 - y2
        p = np.where(residue > 0)[0]; 
        if len(p) > 0:
            yp = residue[p]
            ax.bar(p, yp, color=colors[0], label=str(len(p)))
        
        n = np.where(residue < 0)[0]; 
        if len(n) > 0:
            yn = residue[n]
            ax.bar(n, yn, color=colors[1], label=str(len(n)))
        
        ax.axhline(0, color="b", linewidth=1, linestyle='--')  
        ax.set_ylabel(measures[i])
        ax.legend(prop={'size': 5}, markerscale=0.2,loc='upper right')
        plt.subplots_adjust(hspace=0.001)

        method = '--'.join(methods)
        stats = cbook.boxplot_stats(residue)[0]
        vals = [str(stats[key]) for key in metrics]
        write_row(Config.RESIDUAL_FILE, measures[i], method, vals, args)

    handles, labs = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 6})

def simple_plot(results, fig, labels, colors):
    _, num_datasets, num_measures = results.shape
    x = range(num_datasets)

    for i in range(num_measures):
        ax = fig.add_subplot(num_measures, 1, i+1)
        for k, result_type in enumerate(labels):
            y = results[k,:,i]
            ax.plot(x, y, label=result_type, color=colors[k], linewidth=1)
            ax.set_ylabel(measures[i])
            plt.subplots_adjust(hspace=0.001)
    
    handles, labs = ax.get_legend_handles_labels()
    fig.legend(handles, labs, loc='upper right', prop={'size': 8})

def bar_plot(results, fig, labels, colors):
    _, num_datasets, num_measures = results.shape
    x = range(num_datasets)
    for i in range(num_measures):
        ax = fig.add_subplot(num_measures, 1, i+1)
        for j in x:
            ys = results[:,j, i]
            for y, col, lab in sorted(zip(ys, colors, labels), reverse=True):
                if j == 0:
                    ax.bar([j], [y], color=col, label=lab)
                else:
                    ax.bar([j], [y], color=col)
                
                plt.subplots_adjust(hspace=0.001)

        ax.set_ylabel(measures[i])
        ax.set_xticks(x, [str(p) for p in x])

    handles, labs = ax.get_legend_handles_labels()
    fig.legend(handles, labs, loc='upper right', prop={'size': 6})

def pie_chart(results, fig, labels, colors, methods, args):
    '''
        Plots pie chart and saves results as .csv
    '''
    num_methods, num_datasets, num_measures = results.shape
    
    metrics = ['RESULT']
    check_file_exists(Config.PIECHART_FILE, metrics)

    for i in range(num_measures):
        ax = fig.add_subplot(2, 3, i+1)
        x = []
        for j in range(num_methods):
            x1 = results[j,:,i]
            comp = np.ones(x1.shape, dtype=bool)
            for k in range(num_methods):
                x2 = results[k,:,i]
                if k != j:
                    if i == 1:
                        comp = np.logical_and(comp, np.less(x1, x2))
                    else:
                        comp = np.logical_and(comp, np.greater(x1, x2))
            x.append(sum(comp))

        total = sum(x)
        if total < num_datasets:
            x.append(num_datasets-total)
            if 'C4' not in list(colors): 
                colors = colors + ('C4',)
                labels = labels + ('Tie',)
            
        patches, texts, autotexts = ax.pie(x, colors=colors, labels=x, autopct='%1.1f%%')
        ax.axis('equal')
        ax.set_title(measures[i])
        texts = [text.get_text() for text in autotexts]
        
        method = '--'.join(methods)
        result = '--'.join(texts)
        write_row(Config.PIECHART_FILE, measures[i], method, [result], args)

    fig.legend(patches, labels, loc='lower right', borderpad=0)


def box_plot(results, fig, labels, colors, methods, args):
    '''
        Plots boxplot and saves results as .csv
    '''
    num_methods, num_datasets, num_measures = results.shape
    
    metrics = ['mean', 'iqr', 'cilo', 'cihi', 'whishi', 'whislo', 'q1', 'med', 'q3']
    check_file_exists(Config.BOXPLOT_FILE, metrics)
    
    medianprops = dict(linestyle='-.', linewidth=0.5, color='blue')

    for i in range(num_measures):
        ax = fig.add_subplot(2, 3, i+1)
        X = list(results[:,:,i])
        bp = ax.boxplot(X,patch_artist=True, medianprops=medianprops, notch=True)   
        for j,box in enumerate(bp['boxes']):
            box.set(facecolor = colors[j])

            stats = cbook.boxplot_stats(X[j])[0]
            vals = [str(stats[key]) for key in metrics]
            write_row(Config.BOXPLOT_FILE, measures[i], methods[j], vals, args)

        ax.set_title(measures[i])
        plt.subplots_adjust(hspace=0.4, wspace=0.4)


    fig.legend(bp['boxes'], labels, loc='lower right',borderpad=0)



def plot(results, args, methods, labels, colors):

    if not os.path.exists(Config.RESULT_DIR):
        os.makedirs(Config.RESULT_DIR)

    fig = plt.figure()
    
    if args.plot == 'simple':
        simple_plot(results, fig, labels, colors)
        dpi=800
    elif args.plot == 'bar':
        bar_plot(results, fig, labels, colors)
        dpi=1000
    elif args.plot == 'pie':
        pie_chart(results, fig, labels, colors, methods, args)
        dpi=None
    elif args.plot == 'residual':
        residual_plot(results, fig, labels, colors, methods, args)
        dpi = 400
    elif args.plot == 'box':
        box_plot(results, fig, labels, colors, methods, args)
        dpi=140

    fig.suptitle('Results on {} {} graphs'.format(results.shape[1], args.plot_split))
    return fig, dpi

def make_plots(results, methods, labels, colors, args):
    plot_path = os.path.join(Config.PLOT_DIR, args.dataset, 
        Config.GRAPH_SUBDIR_FORMAT.format(args.min_size, args.max_size),
        str(args.dim), args.source + '-' + args.target, args.plot_split, args.plot)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    if args.plot=='residual':
        M=2;N=2
    elif args.plot=='box':
        M=len(methods)
        N=M
    else: 
        M=2
        N=len(methods)
    indicies = list(range(len(methods)))
    method_info = list(zip(indicies, methods, labels, colors))

    for i in tqdm(range(M, N+1), desc="plotting"):
        for comparison in itertools.combinations(method_info, i):
            indicies, methods, labels, colors = tuple(zip(*comparison)) #unzips
            comp_results = results[indicies,]
            fig,dpi = plot(comp_results, args, methods, labels, colors)
            plot_name = '--'.join(methods) + '.png'
            fig.savefig(os.path.join(plot_path, plot_name), format='png', dpi=dpi)

def compute_anova(results, methods, args):
    
    metrics =['METHOD2', 'meandiff', 'p-adj', 'lower', 'upper', 'reject']
    check_file_exists(Config.ANOVA_FILE, metrics)
    num_methods, num_graphs, num_measures = results.shape

    for i, measure in enumerate(measures):
        scores = results[:,:,i].reshape(-1,1)
        names = np.asarray(methods)
        names = np.repeat(names, num_graphs)
        mc = MultiComparison(scores,names)
        output = mc.tukeyhsd()
        
        for row in output._results_table.data[1:]:
            method = row[0]
            vals = [str(item) for item in row[1:]]
            write_row(Config.ANOVA_FILE, measure, method, vals, args)

def parse_args():
    parser = basic_parser('Compare results')
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
    parser.add_argument('--plot', default='pie', choices = ['simple', 'bar', 'pie', 'residual', 'box', 'ANOVA'],
                        help='The kind of plot. (default: pie)')
    parser.add_argument('--plot_split', default='all', choices = ['train', 'val', 'test', 'all'],
                        help='which split to compare. (default: all)')
    parser.add_argument('--methods', type=str, default="source,source2target,transformed,target",
                        help='comma seperated type of embeddings you would like to compare')
    parser.add_argument('--dim', required=True, type=int,
                        help='Embedding dimension size to be compared')
    args = parser.parse_args()
    return args

def main(args):
    methods = [method.strip() for method in args.methods.split(',')]
    labels = []; colors = []
    for method in methods:
        if method=="source": # OpenNE 
            labels.append(args.source)
            colors.append('C2')
        elif method=="source2target": # OpenNE -> OpenKE
            labels.append(args.source + '->' + args.target)
            colors.append('C1')
        elif method=="transformed":# OpenNE -> model -> Transformed
            labels.append(method)
            colors.append('C3')
        elif method=="target": #OpenKE
            labels.append(args.target)
            colors.append('C5')

    results = load_results(args, methods, labels)
   
    if args.plot == 'ANOVA':
        compute_anova(results, methods,args)
    else:
        make_plots(results, methods, labels, colors, args)

if __name__ == '__main__':
    main(parse_args())

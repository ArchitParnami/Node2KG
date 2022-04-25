import os


class Config:
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    PROJECT_DATASETS = os.path.join(PROJECT_ROOT, 'datasets')
    MODEL_SAVE = os.path.join(PROJECT_ROOT, 'models', 'saved')
    SPLIT_ROOT = os.path.join(PROJECT_ROOT, 'split')
    RANDOM_SEED = 42
    GRAPH_ROOT_DIRNAME = 'graphs'
    GRAPH_SUBDIR_FORMAT = '{}-{}'
    ADJ_LIST_FILE = 'adjListTrain.txt'
    EMBEDDINGS_FILE = 'embeddings.txt'
    EMBEDDINGS_DIR = 'embeddings'
    ARGS_FILE = 'args.txt'
    MODEL_FILE = 'model.pt'
    LOG_FILE = os.path.join(PROJECT_ROOT, 'scripts', 'LOG.txt')
    PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots')
    DATASET_GRAPH_FILE = 'com-{}.ungraph.txt'
    DATASET_COMMUNITY_FILE = 'com-{}.all.cmty.txt'
    RESULT_DIR = os.path.join(PROJECT_ROOT, 'results')
    BOXPLOT_FILE = 'boxplot.csv'
    PIECHART_FILE = 'piechart.csv'
    RESIDUAL_FILE = 'residuals.csv'
    ANOVA_FILE = 'anova.csv'
    TIMING_FILE = os.path.join(RESULT_DIR, 'timing.csv')

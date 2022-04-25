import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import os

LOG_DIR = 'log'
dim = 8
adj_list = 'comm.txt'   

############# CHANGE YOUR CODE HERE #############
from openne.lap import LaplacianEigenmaps
from openne.graph import Graph
from openne.node2vec import Node2vec
g = Graph()
g.read_adjlist(filename=adj_list)
model = Node2vec(graph=g, path_length=80,
                num_paths=10, dim=dim)
vectors = model.vectors
embeddings = np.zeros((g.G.number_of_nodes(), dim))
i=0
labels = []
for label , embedding in vectors.items():
    embeddings[i, :] = embedding
    i = i+1
    labels.append(label)
# embeddings = model.num_embeddings  # numpy array
########################## END ############################
print(labels)
# save embeddings and labels
emb_df = pd.DataFrame(embeddings)
emb_df.to_csv(LOG_DIR + '/embeddings.tsv', sep='\t', header=False, index=False)

lab_df = pd.Series(labels, name='label')
lab_df.to_frame().to_csv(LOG_DIR + '/node_labels.tsv', index=False, header=False)

# save tf variable
embeddings_var = tf.Variable(embeddings, name='embeddings')
sess = tf.Session()

saver = tf.train.Saver([embeddings_var])
sess.run(embeddings_var.initializer)
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)

# configure tf projector
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = 'embeddings'
embedding.metadata_path = 'node_labels.tsv'

projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

# type "tensorboard --logdir=log" in CMD and have fun :)
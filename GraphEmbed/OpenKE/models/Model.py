import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
from GraphEmbed.scripts.read_embeddings import read_embeddings_OpenNE

class Model(nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()
		self.config = config
		self.batch_h = None
		self.batch_t = None
		self.batch_r = None
		self.batch_y = None
	'''
	def get_positive_instance(self):
		self.positive_h = self.batch_h[0:self.config.batch_size]
		self.positive_t = self.batch_t[0:self.config.batch_size]
		self.positive_r = self.batch_r[0:self.config.batch_size]
		return self.positive_h, self.positive_t, self.positive_r

	def get_negative_instance(self):
		self.negative_h = self.batch_h[self.config.batch_size, self.config.batch_seq_size]
		self.negative_t = self.batch_t[self.config.batch_size, self.config.batch_seq_size]
		self.negative_r = self.batch_r[self.config.batch_size, self.config.batch_seq_size]
		return self.negative_h, self.negative_t, self.negative_r
 	'''
	def get_positive_score(self, score):
		return score[0:self.config.batch_size]

	def get_negative_score(self, score):
		negative_score = score[self.config.batch_size:self.config.batch_seq_size]
		negative_score = negative_score.view(-1, self.config.batch_size)
		negative_score = torch.mean(negative_score, 0)
		return negative_score
	def forward(self):
		raise NotImplementedError
	
	def predict(self):
		raise NotImplementedError
	
	def read_pretrained_embeddings(self):
		embedding_file = self.config.pretrained_embeddings_path
		embeddings = read_embeddings_OpenNE(embedding_file)
		if self.config.cuda:
			embeddings = embeddings.cuda()
		return embeddings

	def set_ent_rel_weights(self, ent_embeddings, rel_embeddings):
		if self.config.pretrained_embeddings:
			weights = self.read_pretrained_embeddings()
			ent_embeddings.weight.data = weights
		else:
			nn.init.xavier_uniform(self.ent_embeddings.weight.data)

		# Explicitly setting relation embedding to 1 for amazon dataset
		rel = torch.ones(self.config.relTotal, self.config.hidden_size)
		if self.config.cuda:
			rel = rel.cuda()
		rel_embeddings.weight.data = rel
		rel_embeddings.requires_grad = False

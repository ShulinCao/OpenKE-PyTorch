import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Model import *
class RESCAL(Model):
	def __init__(self,config):
		super(RESCAL,self).__init__(config)
		self.ent_embeddings=nn.Embedding(self.config.entTotal,self.config.hidden_size)
		self.rel_matrices=nn.Embedding(self.config.relTotal,self.config.hidden_size*self.config.hidden_size)
		self.init_weights()
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_matrices.weight.data)
	def _calc(self,h,t,r):
		return h*torch.matmul(r,t)
	def loss_func(self,p_score,n_score):
		criterion= nn.MarginRankingLoss(self.config.margin,False)
		y=Variable(torch.Tensor([1]))
		loss=criterion(p_score,n_score,y)
		return loss
	def forward(self):
		pos_h,pos_t,pos_r=self.get_postive_instance()
		neg_h,neg_t,neg_r=self.get_negtive_instance()
		p_h=self.ent_embeddings(pos_h).view(-1,self.config.hidden_size,1)
		p_t=self.ent_embeddings(pos_t).view(-1,self.config.hidden_size,1)
		p_r=self.rel_matrices(pos_r).view(-1,self.config.hidden_size,self.config.hidden_size)
		n_h=self.ent_embeddings(neg_h).view(-1,self.config.hidden_size,1)
		n_t=self.ent_embeddings(neg_t).view(-1,self.config.hidden_size,1)
		n_r=self.rel_matrices(neg_r).view(-1,self.config.hidden_size,self.config.hidden_size)
		_p_score = self._calc(p_h, p_t, p_r).view(-1, 1, self.config.hidden_size)
		_n_score = self._calc(n_h, n_t, n_r).view(-1, 1, self.config.hidden_size)
		p_score=torch.sum(torch.mean(_p_score,1,False),1)
		n_score=torch.sum(torch.mean(_n_score,1,False),1)
		loss=self.loss_func(p_score,n_score)
		return loss

	def save_parameters(self):
		fent_emb = file("ent_embedding2vec.vec", "wb")
		frel_mat=file("rel_matrices2vec.vec","wb")
		cnt=0
		for param in self.parameters():
			if cnt==0:
				np.savetxt(fent_emb,param.data.numpy(),fmt='%.6f\t')
				cnt=1
			elif cnt==1:
				np.savetxt(frel_mat,param.data.numpy(),fmt='%.6f\t')
				cnt=2
		fent_emb.close()
		frel_mat.close()
		
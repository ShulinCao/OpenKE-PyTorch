import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Model import *
class DistMult(Model):
	def __init__(self,config):
		super(DistMult,self).__init__(config)
		self.ent_embeddings=nn.Embedding(self.config.entTotal,self.config.hidden_size)
		self.rel_embeddings=nn.Embedding(self.config.relTotal,self.config.hidden_size)
		self.softplus=nn.Softplus()
		self.init_weights()
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_embeddings.weight.data)
		
	def _calc(self,h,t,r):
		return torch.sum(h*t*r,1,False)
	def loss_func(self,loss,regul):
		return loss+self.config.lmbda*regul
	def forward(self):
		batch_h,batch_t,batch_r=self.get_all_instance()
		batch_y=self.get_all_labels()
		e_h=self.ent_embeddings(batch_h)
		e_t=self.ent_embeddings(batch_t)
		e_r=self.rel_embeddings(batch_r)
		y=batch_y
		res=self._calc(e_h,e_t,e_r)
		tmp=self.softplus(- y * res)
		loss = torch.mean(tmp)
		regul = torch.mean(e_h ** 2) + torch.mean(e_t ** 2) + torch.mean(e_r ** 2)
		#Calculating loss to get what the framework will optimize
		loss =  self.loss_func(loss,regul)
		return loss
	def save_parameters(self):
		fent_emb = file("ent_embedding2vec.vec", "wb")
		frel_emb=file("rel_embedding2vec.vec","wb")
		cnt=0
		for param in self.parameters():
			if cnt==0:
				np.savetxt(fent_emb,param.data.numpy(),fmt='%.6f\t')
				cnt=1
			elif cnt==1:
				np.savetxt(frel_emb,param.data.numpy(),fmt='%.6f\t')
				cnt=2
		fent_emb.close()
		frel_emb.close()
		
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Model import *
class ComplEx(Model):
	def __init__(self,config):
		super(ComplEx,self).__init__(config)
		self.ent_re_embeddings=nn.Embedding(self.config.entTotal,self.config.hidden_size)
		self.ent_im_embeddings=nn.Embedding(self.config.entTotal,self.config.hidden_size)
		self.rel_re_embeddings=nn.Embedding(self.config.relTotal,self.config.hidden_size)
		self.rel_im_embeddings=nn.Embedding(self.config.relTotal,self.config.hidden_size)
		self.softplus=nn.Softplus()
		self.init_weights()
	def init_weights(self):
		nn.init.xavier_uniform(self.ent_re_embeddings.weight.data)
		nn.init.xavier_uniform(self.ent_im_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_re_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_im_embeddings.weight.data)
	def _calc(self,e_re_h,e_im_h,e_re_t,e_im_t,r_re,r_im):
		return torch.sum(r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t + r_im * e_re_h * e_im_t - r_im * e_im_h * e_re_t,1,False)
	def loss_func(self,loss,regul):
		return loss+self.config.lmbda*regul
	def forward(self):
		batch_h,batch_t,batch_r=self.get_all_instance()
		batch_y=self.get_all_labels()
		e_re_h=self.ent_re_embeddings(batch_h)
		e_im_h=self.ent_im_embeddings(batch_h)
		e_re_t=self.ent_re_embeddings(batch_t)
		e_im_t=self.ent_im_embeddings(batch_t)
		r_re=self.rel_re_embeddings(batch_r)
		r_im=self.rel_im_embeddings(batch_r)
		y=batch_y
		res=self._calc(e_re_h,e_im_h,e_re_t,e_im_t,r_re,r_im)
		tmp=self.softplus(- y * res)
		loss = torch.mean(tmp)
		regul= torch.mean(e_re_h**2)+torch.mean(e_im_h**2)+torch.mean(e_re_t**2)+torch.mean(e_im_t**2)+torch.mean(r_re**2)+torch.mean(r_im**2)
		#Calculating loss to get what the framework will optimize
		loss =  self.loss_func(loss,regul)
		return loss
	def save_parameters(self):
		fent_re=open("entity_re2vec.vec","wb")
		fent_im=open("entity_im2vec.vec","wb")
		frel_re=open("relation_re2vec.vec","wb")
		frel_im=open("relation_im2vec.vec","wb")
		cnt=0
		for param in self.parameters():
			if cnt==0:
				np.savetxt(fent_re,param.data.numpy(),fmt='%.6f\t')
				cnt=1
			elif cnt==1:
				np.savetxt(fent_im,param.data.numpy(),fmt='%.6f\t')
				cnt=2
			elif cnt==2:
				np.savetxt(frel_re,param.data.numpy(),fmt='%.6f\t')
				cnt=3
			elif cnt==3:
				np.savetxt(frel_im,param.data.numpy(),fmt='%.6f\t')
		fent_re.close()
		fent_im.close()
		frel_re.close()
		frel_im.close()
	
	
	
	
		

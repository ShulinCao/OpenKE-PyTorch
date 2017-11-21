#coding:utf-8
import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import datetime
import ctypes

class Config(object):
	def __init__(self):
		self.lib = ctypes.cdll.LoadLibrary("./release/Base.so")
		self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
		self.in_path = "./"
		self.out_path = "./"
		self.bern = 1
		self.hidden_size = 100
		self.ent_size = self.hidden_size
		self.rel_size = self.hidden_size
		self.train_times = 1000
		self.margin = 1.0
		self.nbatches = 100
		self.negative_ent = 1
		self.negative_rel = 0
		self.workThreads = 1
		self.alpha = 0.01
		self.lmbda = 0.000
		self.log_on = 1
		self.lr_decay=0.000
		self.weight_decay=0.000
		self.exportName = None
		self.importName = None
		self.export_steps = 1
		self.optimizer = "SGD"
	def init(self):
		self.lib.setInPath(ctypes.create_string_buffer(self.in_path, len(self.in_path) * 2))
		self.lib.setOutPath(ctypes.create_string_buffer(self.out_path, len(self.out_path) * 2))
		self.lib.setBern(self.bern)
		self.lib.setWorkThreads(self.workThreads)
		self.lib.randReset()
		self.lib.importTrainFiles()
		self.relTotal = self.lib.getRelationTotal()
		self.entTotal = self.lib.getEntityTotal()
		self.tripleTotal = self.lib.getTripleTotal()
		self.batch_size = self.lib.getTripleTotal() / self.nbatches
		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
		self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
		self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
		self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
		self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
		self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
		self.batch_y_addr = self.batch_y.__array_interface__['data'][0]
	def set_lmbda(self, lmbda):
		self.lmbda = lmbda
	def set_optimizer(self, optimizer):
		self.optimizer = optimizer
	def set_log_on(self, flag):
		self.log_on = flag
	def set_alpha(self, alpha):
		self.alpha = alpha
	def set_in_path(self, path):
		self.in_path = path
	def set_out_path(self, path):
		self.out_path = path
	def set_bern(self, bern):
		self.bern = bern
	def set_dimension(self, dim):
		self.hidden_size = dim
		self.ent_size = dim
		self.rel_size = dim
	def set_ent_dimension(self, dim):
		self.ent_size = dim
	def set_rel_dimension(self, dim):
		self.rel_size = dim
	def set_train_times(self, times):
		self.train_times = times
	def set_nbatches(self, nbatches):
		self.nbatches = nbatches
	def set_margin(self, margin):
		self.margin = margin
	def set_work_threads(self, threads):
		self.workThreads = threads
	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate
	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate
	def set_import_files(self, path):
		self.importName = path
	def set_export_files(self, path):
		self.exportName = path
	def set_export_steps(self, steps):
		self.export_steps = steps
	def set_lr_decay(self,lr_decay):
		self.lr_decay=lr_decay
	def set_weight_decay(self,weight_decay):
		self.weight_decay=weight_decay
	def sampling(self):
		self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)
	def set_model(self,model):
		self.model=model
	def train(self):
		trainmodel=self.model(self)
		if self.optimizer == "Adagrad" or self.optimizer == "adagrad":
			optimizer = optim.Adagrad(trainmodel.parameters(), lr=self.alpha,lr_decay=self.lr_decay,weight_decay=self.weight_decay)
		elif self.optimizer == "Adadelta" or self.optimizer == "adadelta":
			optimizer = optim.Adadelta(trainmodel.parameters(), lr=self.alpha)
		elif self.optimizer == "Adam" or self.optimizer == "adam":
			optimizer = optim.Adam(trainmodel.parameters(), lr=self.alpha)
		else:
			optimizer = optim.SGD(trainmodel.parameters(), lr=self.alpha)
		fout=open("loss.txt","wb")
		for epoch in range(self.train_times):
			for batch in range(self.nbatches):
				self.sampling()
				optimizer.zero_grad()
				loss=trainmodel()
				if self.log_on==1:
					print loss.data[0]
				fout.write("%.6f\n"%loss.data[0])
				loss.backward()
				optimizer.step()
		fout.close()
		trainmodel.save_parameters()

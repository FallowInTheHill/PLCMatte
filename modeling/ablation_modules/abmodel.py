import logging
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from torch.nn import functional as F
from modeling import MattingCriterion
from modeling.ablation_modules.ablation_evitmatte import ABEViTMatte


class ABModel(nn.Module):
	def __init__(self, num_heads = 12, embed_dim = 768):
		super().__init__()
		self.evitmatte = ABEViTMatte(num_heads=num_heads, embed_dim=embed_dim)
		self.losses = MattingCriterion(
			losses = ['unknown_l1_loss', 
					  'known_l1_loss', 
					  'loss_pha_laplacian', 
					  'loss_gradient_penalty',
					  ]
					)
		
	def load_dict(self, dir='C:/Users/admin/Desktop/ViTMatte-main/ViTMatte_B_Com.pth'):
		vit_dict = torch.load(dir)
		self_vit_dict = self.evitmatte.state_dict()
		for k, v in vit_dict.items():
			if k in self_vit_dict:
				if v.shape == self_vit_dict[k].shape:
					self_vit_dict.update({k:v})
				elif k == 'backbone.patch_embed.proj.weight':
					single_chanenl_new = v.mean(dim=1, keepdim=True)
					new_channel = torch.cat([single_chanenl_new for i in range(4)],dim=1)
					new_channel = torch.cat([v, new_channel],dim=1)
					self_vit_dict.update({k:new_channel})

		self.evitmatte.load_state_dict(self_vit_dict,strict=False)

	def frozen(self):
		print(f'\nbackbone froze')
		self.evitmatte.backbone.eval()
		self.evitmatte.backbone.requires_grad_(False)
		# self.evitmatte.backbone.pos_embed.requires_grad = False
		# for i in range(len(self.evitmatte.backbone.blocks)):
		# 	m = self.evitmatte.backbone.blocks[i]
		# 	m.eval()
		# 	for name, module in m.named_modules():
		# 		if name == '':
		# 			continue
		# 		if 'mona' not in name:
		# 			module.eval()
		# 			module.requires_grad_(False)
		# 		else:
		# 			module.train()

	def train(self, mode=True, froze = False):
		super(ABModel,self).train(mode)
		if froze:
			self.frozen()
	
	def forward(self, batched_inputs):
		vit_out = self.evitmatte(batched_inputs)
		return vit_out
	
	
if __name__ == '__main__':
	logging.basicConfig(filename='./log/structure_mymodel.log',filemode='w',
					 format='%(message)s',
					 level=logging.INFO)
	# virdict = torch.load('C:/Users/admin/Desktop/ViTMatte-main/ViTMatte_B_Com.pth')
	# model = ABModel()
	# sdict = model.state_dict()
	sdict = torch.load('./pth/EVit-v6-B-schedule4_epoch00124.pth')
	for k in sdict.keys():
		logging.info(f'{k} {sdict[k].shape}')

	
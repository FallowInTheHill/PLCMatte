import logging
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from torch.nn import functional as F
from modeling import PLCMatte, MattingCriterion


class Model(nn.Module):
	def __init__(self, num_heads = 12, embed_dim = 768):
		super().__init__()
		self.plcmatte = PLCMatte(num_heads=num_heads, embed_dim=embed_dim)
		self.losses = MattingCriterion(
			losses = ['unknown_l1_loss', 
					  'known_l1_loss', 
					  'loss_pha_laplacian', 
					  'loss_gradient_penalty',
					  ]
					)

	def load_vitB_dict(self, path = './mae_pretrain_vit_base.pth'):
			vit_dict = torch.load(path)
			self_dict = self.plcmatte.backbone.state_dict()
			for k,v in self_dict.items():
				if k in vit_dict['model']:
					if v.shape == vit_dict['model'][k].shape:
						self_dict.update({k: vit_dict['model'][k]})
					if k == 'patch_embed.proj.weight':
						new_channel = torch.zeros(768,4,16,16)
						new_channel[:,:3] = vit_dict['model'][k]
						self_dict.update({k:new_channel})
			self.plcmatte.backbone.load_state_dict(self_dict)

	def forward(self, batched_inputs):
		vit_out = self.plcmatte(batched_inputs)
		return vit_out
	

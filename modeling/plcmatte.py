import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as F2
import torchvision
import os
from functools import partial
import cv2
from modeling.modules import *
import numpy as np
from thop import profile
from PIL import Image

# embed_dim, num_heads = 768, 12

class PLCMatte(nn.Module):
	def __init__(self,
				 *,
				embed_dim = 768, 
				num_heads =  12,
				pixel_mean = [123.675 / 255., 116.280 / 255., 103.530 / 255.],
				pixel_std = [58.395 / 255., 57.120 / 255., 57.375 / 255.],
				
				 ):
		super(PLCMatte, self).__init__()
		self.backbone =  ViT(
							in_chans=4,
							img_size=512,
							patch_size=16,
							embed_dim=embed_dim,
							depth=12,
							num_heads=num_heads,
							drop_path_rate=0,
							window_size=14,
							mlp_ratio=4,
							qkv_bias=True,
							norm_layer=partial(nn.LayerNorm, eps=1e-6),
							window_block_indexes=[
								# 2, 5, 8 11 for global attention
								0,
								1,
								3,
								4,
								6,
								7,
								9,
								10,
							],
							residual_block_indexes=[2, 5, 8, 11],
							use_rel_pos=True,
							out_feature="last_feat",
						)

		self.decoder =  Classic_Decoder(
					in_chans = embed_dim,
				)
		self.register_buffer(
			"pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
		)
		self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
		assert (
			self.pixel_mean.shape == self.pixel_std.shape
		), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
	
	@property
	def device(self):
		return self.pixel_mean.device

	def forward(self, batched_inputs):
		self.images, self.H, self.W = self.preprocess_inputs(batched_inputs)

		vit_feats = self.backbone(self.images)

		outputs = self.decoder(vit_feats, self.images)  


		outputs['vit_feats'] = vit_feats
		outputs['phas'] = outputs['phas'][:,:,:self.H,:self.W]
		return outputs

	def preprocess_inputs(self, batched_inputs):
		"""
		Normalize, pad and batch the input images.
		"""
		images = batched_inputs["image"].to(self.device)
		trimap = batched_inputs['trimap'].to(self.device)
		images = (images - self.pixel_mean) / self.pixel_std

		image_tri = torch.cat((images, trimap), dim=1)
		
		B, C, self.H, self.W = image_tri.shape
		if image_tri.shape[-1]%32!=0 or image_tri.shape[-2]%32!=0:
			new_H = (32-image_tri.shape[-2]%32) + self.H
			new_W = (32-image_tri.shape[-1]%32) + self.W
			new_images = torch.zeros((image_tri.shape[0], image_tri.shape[1], new_H, new_W)).to(self.device)
			new_images[:,:,:self.H,:self.W] = image_tri[:,:,:,:]
			del image_tri
			image_tri = new_images

		return image_tri, self.H, self.W

	
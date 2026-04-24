import torch
from torch import nn
from torch.nn import functional as F
from modeling.utils import LN

class Basic_Conv3x3(nn.Module):
	"""
	Basic convolution layers including: Conv3x3, BatchNorm2d, GELU layers.
	"""
	def __init__(
		self,
		in_chans,
		out_chans,
		stride=2,
		padding=1,
	):
		super().__init__()
		self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
		self.bn = nn.BatchNorm2d(out_chans)
		self.act = nn.GELU()

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.act(x)

		return x

class Fusion_Block(nn.Module):
	"""
	Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
	"""
	def __init__(
		self,
		in_chans,
		out_chans,
	):
		super().__init__()
		self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

	def forward(self, x, D):
		F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
		out = torch.cat([D, F_up], dim=1)
		out = self.conv(out)

		return out	

class Matting_Head(nn.Module):
	"""
	Simple Matting Head, containing only conv3x3 and conv1x1 layers.
	"""
	def __init__(
		self,
		in_chans = 32,
		mid_chans = 16,
	):
		super().__init__()
		self.matting_convs = nn.Sequential(
			nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
			nn.BatchNorm2d(mid_chans),
			nn.GELU(),
			nn.Conv2d(mid_chans, 1, 1, 1, 0)
			)

	def forward(self, x):
		x = self.matting_convs(x)

		return x

class Decoder_FHalf(nn.Module):
	"""
	Simple and Lightweight Detail Capture Module for ViT Matting.
	"""
	def __init__(
		self,
		in_chans = 768,
		
		# pyramid_out = [48, 96, 192, 384],
	):
		super().__init__()
		self.feats_pyramid = SimpleFeaturePyramid(scale_factors=[8], dim=in_chans)
		

	def forward(self, features):
		deep_vit_features = self.feats_pyramid(features)[0]
		# shallow_conv_features = self.convstream(images)
		# fusion_feats = self.fusion_neck(deep_vit_features,shallow_conv_features)
		
		# phas = torch.sigmoid(self.matting_head(fusion_feats))

		return deep_vit_features

class SimpleFeaturePyramid(nn.Module):
	"""
	This module implements SimpleFeaturePyramid in :paper:`vitdet`.
	It creates pyramid features built on top of the input feature map.
	"""

	def __init__(
		self,
		scale_factors=[2,4,8,16],
		use_bias = True,
		dim = 768
	):

		super(SimpleFeaturePyramid, self).__init__()

		self.scale_factors = scale_factors

		# input_shapes = net.output_shape()

		self.stages = nn.ModuleList()
		for idx, scale in enumerate(scale_factors):
			out_dim = dim
			if scale == 8.0:
				layers = [
					nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
					LN(dim//2),
					nn.GELU(),
					nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
					LN(dim//4),
					nn.GELU(),
					nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
				]
				out_dim = dim // 8
			
			else:
				raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

			layers.extend(
				[
					nn.Conv2d(
						out_dim,
						out_dim,
						kernel_size=1,
						bias=use_bias,
					),
					LN(out_dim),
					nn.Conv2d(
						out_dim,
						out_dim,
						kernel_size=3,
						padding=1,
						bias=use_bias,
					),
					LN(out_dim),
				]
			)
			layers = nn.Sequential(*layers)
			self.stages.append(layers)
			self.out_dim = out_dim

	def forward(self, x):
		results = []
		for stage in self.stages:
			results.append(stage(x))
		return results

class Decoder_SHalf(nn.Module):
	def __init__(self, 
			     fusion_out = [32],
			     in_chans = 768,):
		super().__init__()
		self.fusion_out = fusion_out
		self.matting_head = Matting_Head(
			in_chans = fusion_out[-1],
		)
		self.convstream = Basic_Conv3x3(4,in_chans//8,stride=1)
		self.fusion_neck = Fusion_Block(in_chans//8+in_chans//8,fusion_out[-1])
	def forward(self, images, x):
		deep_vit_features = x
		shallow_conv_features = self.convstream(images)
		fusion_feats = self.fusion_neck(deep_vit_features,shallow_conv_features)
		phas = torch.sigmoid(self.matting_head(fusion_feats))

		return {
				'phas': phas,
				'deep_vit_feats': deep_vit_features, 
				# 'vit_feats': features,
				# 'fusion_feats': fusion_feats
				}
		pass


if __name__ == '__main__':
	from thop import profile

	# fea = torch.zeros(1,768,32,32)
	# images = torch.zeros(1,4,512,512)
	# M = Classic_Decoder_FHalf(in_chans=768)
	# out = M(fea,images)
	# print(out['phas'].shape)
	# flops, paramas = profile(M,(fea,images))
	# print(f'FLOPs {flops} , paramas {paramas}')
	#FLOPs 37,369,151,488.0 , paramas 1,706,529.0
	#FLOPs 332,117,573,632.0 , paramas 4,875,297.0

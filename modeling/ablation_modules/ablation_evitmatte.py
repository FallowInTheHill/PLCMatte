import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as F2
import torchvision
import os
from functools import partial
import cv2
from modeling.modules import *
from modeling.ablation_modules.separated_decoders import *
import numpy as np
from thop import profile
from PIL import Image

# embed_dim, num_heads = 768, 12

class ABEViTMatte(nn.Module):
	def __init__(self,
				 *,
				embed_dim = 384, 
				num_heads =  6,
				# criterion = None,
				pixel_mean = [123.675 / 255., 116.280 / 255., 103.530 / 255.],
				pixel_std = [58.395 / 255., 57.120 / 255., 57.375 / 255.],
				# input_format = "RGB",
				
				 ):
		super(ABEViTMatte, self).__init__()
		self.backbone =  EViT(  # Single-scale ViT backbone
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

		self.decoder_fhalf =  Decoder_FHalf(
					in_chans = embed_dim,
				)
		self.decoder_shalf = Decoder_SHalf()
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
		with torch.no_grad():
			self.images, self.H, self.W, gt = self.preprocess_inputs(batched_inputs)

			vit_feats = self.backbone(self.images)
			deepvit_feats = self.decoder_fhalf(vit_feats) 
			shaped_dvf = self.midprocess_feats(deepvit_feats, gt)
		outputs = self.decoder_shalf(self.images, shaped_dvf) 

		outputs['vit_feats'] = vit_feats
		outputs['phas'] = outputs['phas'][:,:,:self.H,:self.W]
		return outputs

	def preprocess_inputs(self, batched_inputs):
		"""
		Normalize, pad and batch the input images.
		"""
		images = batched_inputs["image"].to(self.device)
		trimap = batched_inputs['trimap'].to(self.device)
		gt = batched_inputs['gt'].to(self.device)
		images = (images - self.pixel_mean) / self.pixel_std

		# if 'fg' in batched_inputs.keys():
		# 	trimap[trimap < 85] = 0
		# 	trimap[trimap >= 170] = 1
		# 	trimap[trimap >= 85] = 0.5

		image_tri = torch.cat((images, trimap), dim=1)
		
		B, C, self.H, self.W = image_tri.shape
		if image_tri.shape[-1]%32!=0 or image_tri.shape[-2]%32!=0:
			new_H = (32-image_tri.shape[-2]%32) + self.H
			new_W = (32-image_tri.shape[-1]%32) + self.W
			new_images = torch.zeros((image_tri.shape[0], image_tri.shape[1], new_H, new_W)).to(self.device)
			new_images[:,:,:self.H,:self.W] = image_tri[:,:,:,:]
			del image_tri
			image_tri = new_images

		return image_tri, self.H, self.W, gt
	def midprocess_feats(self, dvf, gt):
		with torch.no_grad():
			gt = F.interpolate(gt,dvf.shape[-2:],mode='bilinear')
			unknow_mask = ((gt>0)&(gt<1)).float() #B,1,H,W ((gt>0))表现不佳
			# unknow_mask_sum = unknow_mask.sum(dim=(-2,-1),keepdim=True)+1e-3 #B,1,1,1
			active_unknow_mask = unknow_mask * (dvf>0).float()	##B,C,H,W
			active_unknow_mask_sum = active_unknow_mask.sum(dim=(-2,-1),keepdim=True) + 1e-3

			active_f_mean = (dvf*active_unknow_mask).sum(dim=(-2,-1),keepdim=True) / active_unknow_mask_sum
			active_f_var =(((dvf - active_f_mean)*active_unknow_mask).square()).sum(dim=(-2,-1),keepdim=True) / active_unknow_mask_sum
			active_gt_mean = ((gt*active_unknow_mask).sum(dim=(-2,-1),keepdim=True) / active_unknow_mask_sum)
			active_gt_var = (((gt - active_gt_mean)*active_unknow_mask).square()).sum(dim=(-2,-1),keepdim=True) / active_unknow_mask_sum + 1e-11
			gt_standard2 = (gt / torch.sqrt(active_gt_var) * torch.sqrt(active_f_var))*active_unknow_mask
			gt_standard2 = (gt_standard2 - (gt_standard2.sum(dim=(-2,-1),keepdim=True)/active_unknow_mask_sum) + active_f_mean)*active_unknow_mask

		return gt_standard2
def test_inference():
	# image = torch.zeros([1,3,512,512])
	outdir = 'tempresult-3'
	image = F2.to_tensor(Image.open('C:/Users/admin/Desktop/ViTMatte-main/demo/raw-16452523375_08591714cf_o_16.png').convert('RGB')).unsqueeze(0)
	# trimap = torch.zeros([1,1,512,512])
	trimap = F2.to_tensor(Image.open('C:/Users/admin/Desktop/ViTMatte-main/demo/trimap-16452523375_08591714cf_o_16.png').convert('L')).unsqueeze(0)
	
	input = {'image':image, 'trimap':trimap}
	V = ABEViTMatte()
	V.load_state_dict(torch.load('C:/Users/admin/Desktop/ViTMatte-main/ViTMatte_B_Com.pth'))
	V.eval()
	with torch.no_grad():
		outs = V(input)
		print(outs['features'])
		# resize_size = (outs['features'].shape[-1],outs['features'].shape[-2])
		# tri_resize = trimap.clone().squeeze().numpy()
		# tri_resize = cv2.resize(tri_resize,resize_size)
		# tri_resize = torch.from_numpy(tri_resize)
		# mask = torch.zeros(tri_resize.shape)
		# mask[tri_resize>0] = 1
		# feas_new=[]
		# cnt = 0
		# for item in outs['features'][0]:
		# 	Image.fromarray(item.numpy()*255).convert('L').save(f'C:/Users/admin/Desktop/ViTMatte-main/{outdir}/{cnt}.jpg')
		# 	tf = item*mask
		# 	Image.fromarray(tf.numpy()*255).convert('L').save(f'C:/Users/admin/Desktop/ViTMatte-main/{outdir}/d_{cnt}.jpg')
		# 	cnt+=1
		# 	tf = tf.unsqueeze(0)
		# 	feas_new.append(tf)
		# feas_new = torch.cat(feas_new,0).unsqueeze(0)
		# F2.to_pil_image(d_out['phas'][:,:,:V.H,:V.W].to('cpu').flatten(0, 2)).save(f'C:/Users/admin/Desktop/ViTMatte-main/{outdir}/d_alpha.jpg')# print(outs)
		# F2.to_pil_image(outs['phas'].to('cpu').flatten(0, 2)).save(f'C:/Users/admin/Desktop/ViTMatte-main/{outdir}/alpha.jpg')# print(outs)
		# for idx in range(outs['features'].shape[1]):
		# 	feas_new = outs['features'].clone()
		# 	feas_new[:,idx,:,:] = torch.zeros(feas_new[:,0,:,:].shape)
		# 	d_out = V.decoder(feas_new,V.images)
		# 	F2.to_pil_image(d_out['phas'][:,:,:V.H,:V.W].to('cpu').flatten(0, 2)).save(f'C:/Users/admin/Desktop/ViTMatte-main/{outdir}/alpha_{idx}.jpg')# print(outs)


if __name__ == '__main__':
	
	model = ABEViTMatte()
	dicts = model.state_dict()
	for item in dicts.items():
		print(f'{item[0]}	{item[1].shape}')
	# input = {'image':torch.randn(1, 3, 512, 512), 'trimap':torch.randn(1, 1, 512, 512)}
	# out = model(input)
	# flops, params = profile(model, inputs=(input,))
	# print(f"FLOPs: {flops/1e9} G | Params: {params/1e6} M")  # 输出结果[1,3](@ref)

	
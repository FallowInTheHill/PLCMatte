
def nlc_to_nchw(x, hw_shape):
	"""Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

	Args:
		x (Tensor): The input tensor of shape [N, L, C] before conversion.
		hw_shape (Sequence[int]): The height and width of output feature map.

	Returns:
		Tensor: The output tensor of shape [N, C, H, W] after conversion.
	"""
	H, W = hw_shape
	assert len(x.shape) == 3
	B, L, C = x.shape
	assert L == H * W, 'The seq_len doesn\'t match H, W'
	return x.transpose(1, 2).reshape(B, C, H, W)
def nchw_to_nlc(x):
	"""Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

	Args:
		x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

	Returns:
		Tensor: The output tensor of shape [N, L, C] after conversion.
	"""
	assert len(x.shape) == 4
	return x.flatten(2).transpose(1, 2).contiguous()


import os
import shutil
import torch
import numpy as np
import cv2
import random
from PIL import Image
#from config import *

##########################
### Pure functions
##########################

def cycle(iterable):
	while True:
		for x in iterable:
			yield x

def extract_pure_name(original_name):
	pure_name, extention = os.path.splitext(original_name)
	return pure_name
def listdir_nohidden(path):

	#无前缀
	new_list = []
	for f in os.listdir(path):
		if not f.startswith('.'):
			new_list.append(f)
	new_list.sort()
	return new_list

def create_folder_if_not_exists(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

def check_if_folder_exists(folder_path):
	return os.path.exists(folder_path)
	

def refresh_folder(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	else:
		shutil.rmtree(folder_path)
		os.makedirs(folder_path)

def generate_composite_img(img, alpha_channel):
	b_channel, g_channel, r_channel = cv2.split(img)
	b_channel = b_channel * alpha_channel
	g_channel = g_channel * alpha_channel
	r_channel = r_channel * alpha_channel
	alpha_channel = (alpha_channel*255).astype(b_channel.dtype)
	img_BGRA = cv2.merge((r_channel,g_channel,b_channel,alpha_channel))
	return img_BGRA

def extract_pure_fg(img, alpha_channel):
	alpha_channel = alpha_channel.astype(np.float32)/255
	r_channel, g_channel, b_channel = img[:,:,0],img[:,:,1],img[:,:,2]

	b_channel = b_channel * alpha_channel
	g_channel = g_channel * alpha_channel
	r_channel = r_channel * alpha_channel
	# alpha_channel = (alpha_channel*255).astype(b_channel.dtype)
	img_A = np.dstack((
					r_channel.astype(np.uint8),
					g_channel.astype(np.uint8),
					b_channel.astype(np.uint8),
					(alpha_channel*255).astype(np.uint8)
					))
	return img_A

def merge_2_fgs(im1,im2):
	#im1盖住im2
	r1, g1, b1, a1 = im1[...,0], im1[...,1], im1[...,2], (im1[...,3].astype(np.float32)/255)
	r2, g2, b2, a2 = im2[...,0], im2[...,1], im2[...,2], (im2[...,3].astype(np.float32)/255)
	
	# 3. 计算混合后的颜色和透明度
	alpha_factor = 1.0 - a1  # im1的剩余透明度权重
	blended_r = (r1 * a1) + (r2 * a2 * alpha_factor)
	blended_g = (g1 * a1) + (g2 * a2 * alpha_factor)
	blended_b = (b1 * a1) + (b2 * a2 * alpha_factor)
	blended_a = a1 + (a2 * alpha_factor)
	
	# 4. 反归一化并合并通道
	blended_rgba = np.dstack((
		np.clip(blended_r, 0, 255).astype(np.uint8),
		np.clip(blended_g, 0, 255).astype(np.uint8),
		np.clip(blended_b, 0, 255).astype(np.uint8),
		np.clip(blended_a*255, 0, 255).astype(np.uint8)
	))
	
	return blended_rgba, np.clip(blended_a*255, 0, 255).astype(np.uint8)

def merge_fgbg(fg,alpha,bg):
	#im1盖住im2
	r1, g1, b1 = fg[...,0], fg[...,1], fg[...,2]
	r2, g2, b2 = bg[...,0], bg[...,1], bg[...,2]
	alpha = alpha.astype(np.float32)/255
	# 3. 计算混合后的颜色和透明度
	alpha_factor = 1.0 - alpha  # im1的剩余透明度权重
	blended_r = (r1 * alpha) + (r2 * alpha_factor)
	blended_g = (g1 * alpha) + (g2 * alpha_factor)
	blended_b = (b1 * alpha) + (b2 * alpha_factor)
	
	
	# 4. 反归一化并合并通道
	blended_rgba = np.dstack((
		np.clip(blended_r, 0, 255).astype(np.uint8),
		np.clip(blended_g, 0, 255).astype(np.uint8),
		np.clip(blended_b, 0, 255).astype(np.uint8),
	))
	
	return blended_rgba



#######################################
### Function to generate training data
#######################################
##### for 一般情况 #####
def generate_paths(PATH = "C:/Users/admin/Desktop/SYN70k/train/blendall/"):
	#获取完整路径
	mask_list = listdir_nohidden(PATH)
	paths_list = []

	for mask_name in mask_list:
		ori_path = PATH + mask_name
		paths_list.append(ori_path)
	return paths_list

def generate_paths_for_vitmatting(PATH = 'E:/dataset/Comp1k_aug_composited/'):
	names = listdir_nohidden(PATH+'img/')
	cast = []
	for name in names:
		cast.append([PATH+'img/'+name, PATH+'alpha/'+name, PATH+'trimap/'+name])
	return cast
def generate_paths_for_compo1k(PATH= ''):
	names = listdir_nohidden(PATH+'merged/')
	cast = []
	for name in names:
		cast.append([PATH+'merged/'+name, PATH+'alpha_copy/'+name, PATH+'trimaps/'+name])
	return cast
def generate_paths_for_dis646(PATH= ''):
	names = listdir_nohidden(PATH+'GT/')
	cast = []
	for name in names:
		cast.append([PATH+'Image/'+name, PATH+'GT/'+name, PATH+'trimap/'+name])
	return cast

def trim_img(img):
	if img.ndim>2:
		img = img[:,:,0]
	return img

def resize_img(ori, img):
	img = cv2.resize(img, ori.shape)*255.0
	return img

def process_fgbg(ori, mask, is_fg, fgbg_path=None):
	if fgbg_path is not None:
		img = np.array(Image.open(fgbg_path))
	else:
		mask_3 = (mask/255.0)[:, :, np.newaxis].astype(np.float32)
		img = ori*mask_3 if is_fg else ori*(1-mask_3)
	return img

# def add_guassian_noise(img, fg, bg):
# 	row,col,ch= img.shape
# 	mean = 0
# 	sigma = 10
# 	gauss = np.random.normal(mean,sigma,(row,col,ch))
# 	gauss = gauss.reshape(row,col,ch)
# 	noisy_img = np.uint8(img + gauss)
# 	noisy_fg = np.uint8(fg + gauss)
# 	noisy_bg = np.uint8(bg + gauss)
# 	return noisy_img, noisy_fg, noisy_bg

# def generate_composite_rssn(fg, bg, mask, fg_denoise=None, bg_denoise=None):
# 	## resize bg accordingly
# 	h, w, c = fg.shape
# 	alpha = np.zeros((h, w, 1), np.float32)
# 	alpha[:, :, 0] = mask / 255.
# 	bg = resize_img(fg, bg)
# 	## use denoise fg/bg randomly
# 	if fg_denoise is not None and random.random()<0.5:
# 		fg = fg_denoise
# 		bg = resize_img(fg, bg_denoise)
# 	## reduce sharpness discrepancy
# 	if random.random()<0.5:
# 		rand_kernel = random.choice([20,30,40,50,60])
# 		bg = cv2.blur(bg, (rand_kernel,rand_kernel))
# 	composite = alpha * fg + (1 - alpha) * bg
# 	composite = composite.astype(np.uint8)
# 	## reduce noise discrepancy
# 	if random.random()<0.5:
# 		composite, fg, bg = add_guassian_noise(composite, fg, bg)
# 	return composite, fg, bg

# def generate_composite_coco(fg, bg, mask):
# 	h, w, c = fg.shape
# 	alpha = np.zeros((h, w, 1), np.float32)
# 	alpha[:, :, 0] = mask / 255.
# 	bg = resize_img(fg, bg)
# 	composite = alpha * fg + (1 - alpha) * bg
# 	composite = composite.astype(np.uint8)
# 	return composite, fg, bg

#data augment
def gen_dilate(alpha, kernel_size): 
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
	fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
	dilate =  cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
	return dilate.astype(np.uint8)

def gen_erosion(alpha, kernel_size): 
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
	fg = np.array(np.equal(alpha, 255).astype(np.float32))
	erode = cv2.erode(fg, kernel, iterations=1)*255
	return erode.astype(np.uint8)

def gen_trimap_from_alpha(alpha, kernel_size=3, iteration=1):	
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))

	fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
	fg = np.array(np.equal(alpha, 255).astype(np.float32))

	dilate =  cv2.dilate(fg_and_unknown, kernel, iterations=iteration)
	erode = cv2.erode(fg, kernel, iterations=iteration)

	trimap = erode *255 + (dilate-erode)*128
	return trimap.astype(np.uint8)

def validate_param_groups(model, optimizer:torch.optim.AdamW):
	"""验证优化器的参数组是否完整且无重复"""
	# --- 检查重复参数 ---
	custom_params = []
	for group in optimizer.param_groups:
		custom_params.extend(group["params"])
	
	seen = set()
	duplicates = []
	for p in custom_params:
		pid = id(p)
		if pid in seen:
			duplicates.append(p)
		else:
			seen.add(pid)
	
	if duplicates:
		print(f"参数重复！重复参数数量: {len(duplicates)}")
	
	# --- 检查遗漏参数 ---
	model_params = set(p for p in model.parameters() if p.requires_grad)
	custom_params_set = set(p for p in custom_params if p.requires_grad)
	missing_params = model_params - custom_params_set
	
	if missing_params:
		raise ValueError(f"参数遗漏！遗漏参数数量: {len(missing_params)}")
	
	print("验证通过：参数组完整覆盖模型参数")

def deduplicate_param_groups(param_groups):
	"""
	去重参数组，保留每个参数最后一次出现的配置
	:param_groups: 原始参数组列表，格式为 [{"params": [p1,p2], "lr": 0.1}, ...]
		先定义的参数被后定义的参数覆盖, 以group的形式组织
	:return: 去重后的参数组列表
	"""
	dedup_groups = []
	seen_pids = set()
	
	for group in reversed(param_groups):
		new_group = {"params": []}
		# 继承当前组的默认配置（可能被后续参数覆盖）
		new_group.update({k: v for k, v in group.items() if k != "params"})
		
		for p in group["params"]:
			pid = id(p)
			if pid not in seen_pids:
				new_group["params"].append(p)
				# 更新为最后一次的配置
				# new_group.update(param_registry[pid])
				seen_pids.add(pid)
		
		if new_group["params"]:
			dedup_groups.append(new_group)
	
	return dedup_groups

def FixSeed():
	# some cudnn methods can be random even after fixing the seed
	# unless you tell it to be deterministic
	seed = 0
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True

def get_TrainStage(milestones, index):
	#返回索引
	epoch_stages = [milestones[r]["epoch"] for r in range(len(milestones))]
	stage = -1
	for i in range(len(epoch_stages)):
		if epoch_stages[i] <= index:
			stage += 1
	assert stage != -1, "Error: Train Stage"
	return stage

if __name__ == '__main__':
	# paths = generate_path_for_test()
	print()
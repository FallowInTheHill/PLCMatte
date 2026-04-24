'''
Inference for Composition-1k Dataset.

Run:
python inference.py \
	--config-dir path/to/config
	--checkpoint-dir path/to/checkpoint
	--inference-dir path/to/inference
	--data-dir path/to/data
'''
import os
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F2
from torch.nn import functional as F
from os.path import join as opj
import numpy as np

from model import Model as PLCMModel
from utils import refresh_folder

import warnings
warnings.filterwarnings('ignore')

#Dataset and Dataloader
def collate_fn(batched_inputs):
	rets = dict()
	for k in batched_inputs[0].keys():
		rets[k] = torch.stack([_[k] for _ in batched_inputs])
	return rets

def sort_files_by_size(directory, reverse=False):
	files = [f for f in os.listdir(directory) 
			 if os.path.isfile(os.path.join(directory, f))]
	files.sort(key=lambda x: os.path.getsize(opj(directory,x)), reverse=reverse)
	return files

class Composition_1k(Dataset):
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.file_names = sort_files_by_size(opj(self.data_dir, 'merged'))

	def __len__(self):
		return len(self.file_names)

	def __getitem__(self, idx):
		phas = Image.open(opj(self.data_dir, 'alpha_copy', self.file_names[idx])).convert('L')
		tris = Image.open(opj(self.data_dir, 'trimaps', self.file_names[idx])).convert('L')
		imgs = Image.open(opj(self.data_dir, 'merged', self.file_names[idx])).convert('RGB')
		sample = {}

		sample['trimap'] = F2.to_tensor(tris)
		sample['image'] = F2.to_tensor(imgs)
		sample['gt'] = F2.to_tensor(phas)
		sample['image_name'] = self.file_names[idx]

		return sample

def matting_inference(
	checkpoint_dir='',
	inference_dir='',
	data_dir='',
	dataset_name = ''
):

	model = PLCMModel()
	model.eval()
	trained_dict = torch.load(checkpoint_dir, map_location='cpu')
	model.load_state_dict(trained_dict)
	model.to(device)

	if dataset_name == 'comp1k':
		dataloader = DataLoader(
			dataset = Composition_1k(
				data_dir = data_dir
			),
			shuffle = False,
			batch_size = 1,
		)
	else:
		print('dataset name error')
		return
	
	#inferencing
	refresh_folder(inference_dir)
	os.makedirs(inference_dir, exist_ok=True)
	for data in tqdm(dataloader):
		with torch.no_grad():
			if os.path.exists(opj(inference_dir, data['image_name'][0])):
				continue
			for k in data.keys():
				if k == 'image_name':
					continue
				else:
					if k == 'trimap':
						data[k][(data[k]>0.1)&(data[k]<0.9)] = 0.5
					data[k].to(device)
			output = model(data)['phas'].flatten(0, 2)
			output[data['trimap'].flatten(0,2)>0.9]=1
			output[data['trimap'].flatten(0,2)<0.1]=0
			output = F2.to_pil_image(output)
			output.save(opj(inference_dir, data['image_name'][0]))
			torch.cuda.empty_cache()

def inference_eval_pearson(
	checkpoint_dir='',
	data_dir='',
	logger = None
):
	model = PLCMModel()
	model.eval()
	trained_dict = torch.load(checkpoint_dir, map_location='cpu')
	model.load_state_dict(trained_dict)
	model.to(device)
	dataloader = DataLoader(
			dataset = Composition_1k(
				data_dir = data_dir
			),
			shuffle = False,
			batch_size = 1,
		)
	pearson_list = np.zeros(96)
	for data in tqdm(dataloader):
		with torch.no_grad():
			for k in data.keys():
				if k == 'image_name':
					continue
				else:
					if k == 'trimap':
						data[k][(data[k]>0.1)&(data[k]<0.9)] = 0.5
					data[k].to(device)
			output = model(data)['deep_vit_feats'].cpu()
			gt = data['gt']
			B,C,H,W = gt.shape
			if gt.shape[-1]%32!=0 or gt.shape[-2]%32!=0:
				new_H = (32-gt.shape[-2]%32) + H
				new_W = (32-gt.shape[-1]%32) + W
				new_gt = torch.zeros((gt.shape[0], gt.shape[1], new_H, new_W)).to(device)
				new_gt[:,:,:H,:W] = gt[:,:,:,:]
				del gt
				gt = new_gt
			gt = F.interpolate(gt,scale_factor=0.5,mode='bilinear', align_corners=False)[0,...].cpu()
			torch.cuda.empty_cache()
			gt.to(device)
			output.to(device)
			one_img_pearson = np.zeros(96)
			for cidx in range(output.shape[1]):
				Feats = output[:,cidx,:,:]
				pearson = pearson_correlation_corrcoef(Feats,gt,((gt>0)&(gt<1)&(Feats>0)))
				one_img_pearson[cidx] = (pearson)
				pearson_list[cidx] += (pearson)
			logger.info(f'{data['image_name']}: {one_img_pearson}')
	logger.info(f'all: {pearson_list/len(dataloader)}')

def pearson_correlation_corrcoef(F, reGT, mask):

	a_flat = F[mask]
	b_flat = reGT[mask]
	
	if len(a_flat) < 2 or len(b_flat) < 2:
		return 'len'

	if torch.isnan(a_flat).any() or torch.isinf(a_flat).any() or torch.isnan(b_flat).any() or torch.isinf(b_flat).any():
		print("NAN or INF in data")
		return 'nan'
	
	if torch.std(a_flat) == 0 or torch.std(b_flat) == 0:
		return 'ZeroStd'

	corr_matrix = torch.corrcoef(torch.stack([a_flat, b_flat]))
	pearson_r = corr_matrix[0, 1]
	return pearson_r.item()

if __name__ == '__main__':
	#add argument we need:
	device = torch.device('cuda')
	parser = {}
	parser['checkpoint_dir'] = './PLCMatte_B.pth'
	parser['inference_dir'] = './CompoRes/'
	parser['data_dir'] = '/Matteformer_Composition1k/Composition-1k-testset/',

	matting_inference(
		checkpoint_dir = parser['checkpoint_dir'],
		inference_dir = parser['inference_dir'],
		data_dir = parser['data_dir'],
		dataset_name = 'comp1k'
	)
	# import logging
	# logger = logging.getLogger("mylogger")
	# logger.setLevel(logging.INFO)
	# formatter = logging.Formatter('%(message)s')
	# file_handler = logging.FileHandler(f"./log/Pearson_BaseModel",mode='w')
	# file_handler.setFormatter(formatter)
	# logger.addHandler(file_handler)
	# inference_eval_pearson(
	# 	checkpoint_dir = parser['checkpoint_dir'],
	# 	data_dir = parser['data_dir'],
	#   logger = logger
	# )
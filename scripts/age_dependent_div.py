import os
import time
import numpy as np
from argparse import Namespace
from tqdm import tqdm
from PIL import Image
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_image
from models.psp import pSp
from criteria import id_loss
from criteria.lpips.lpips import LPIPS
from criteria.aging_loss import AgingLoss
from criteria.earlystopping import EarlyStopping
from torch.optim import Adam
from utils import train_utils
from datasets.augmentations import AgeTransformer


class ADFD:
	def __init__(self, opts):
		self.test_opts = opts

		# update test options with options used during training
		ckpt = torch.load(self.test_opts.checkpoint_path, map_location='cpu')
		opts = ckpt['opts']
		opts.update(vars(self.test_opts))
		self.opts = Namespace(**self.opts)

		self.net = pSp(self.opts)
		self.net.eval()
		self.net.cuda()
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		# Initialize loss
		self.mse_loss = nn.MSELoss().cuda().eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').cuda().eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().cuda().eval()
		if self.opts.aging_lambda > 0:
			self.aging_loss = AgingLoss()
		
	def diversify(self):
		if self.opts.couple_outputs:
			multi_path_results = os.path.join(self.test_opts.exp_dir, 'multimodal_coupled')
		else:
			multi_path_results = os.path.join(self.test_opts.exp_dir, 'multimodal_results')
		os.makedirs(multi_path_results, exist_ok=True)
		
		age_transformers = [AgeTransformer(target_age=age) for age in self.opts.target_age.split(',')]

		print(f'Loading dataset')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		dataset = InferenceDataset(root=self.opts.data_path,
								transform=transforms_dict['transform_inference'],
								opts=self.opts)
		dataloader = DataLoader(dataset,
								batch_size=self.opts.test_batch_size,
								shuffle=False,
								num_workers=int(self.opts.test_workers),
								drop_last=False)

		if self.opts.n_images is None:
			self.opts.n_images = len(dataset)

		for age_transformer in age_transformers:
			print(f"Running on target age: {age_transformer.target_age}")
			global_i = 0
			global_time = []
			for input_batch in tqdm(dataloader):
				if global_i >= self.opts.n_images:
					break
				input_age_batch = [age_transformer(img.cpu()).to('cuda') for img in input_batch]
				input_age_batch = torch.stack(input_age_batch)
				for image_idx, input_image in enumerate(input_age_batch):
					result, sv = self.net(input_image.unsqueeze(0).to("cuda").float(), randomize_noise=False, resize=self.opts.resize_outputs, return_latents=False, return_s=True)
					resize_amount = (256, 256) if self.opts.resize_outputs else (1024, 1024)
					initial_age = self.aging_loss.extract_ages(result)
					
					input_image = input_image[:3].unsqueeze(0).to("cuda").float()
					result = self.face_pool(result)

					im_path = dataset.paths[global_i]
					image_name = os.path.splitext(os.path.basename(im_path))[0]
					image_ext = os.path.splitext(os.path.basename(im_path))[1]
					image = input_batch[image_idx]
					orig_image = log_image(image, self.opts).resize(resize_amount)
					
					if self.opts.couple_outputs:
						multi_modal_outputs = []

					# optimization
					for multi_i in range(self.opts.n_outputs_to_generate):
						tic = time.time()
						s_mod = self.age_id_based_perturbation(sv)
						if self.opts.div_opt == 'adam':
							optimizer = Adam([s for i, s in enumerate(s_mod) if i % 3 != 1], self.opts.div_lr)
						earlystopping = EarlyStopping(patience=self.opts.patience, delta=self.opts.es_delta)
						for step in tqdm(range(self.opts.max_steps)):
							optimizer.zero_grad()
							if step == 0:
								initial_result = result.detach().clone()
								target_ages = initial_age.detach().clone() / 100.
								input_ages = target_ages
							else:
								input_ages = self.aging_loss.extract_ages(y_hat) / 100.

							y_hat = self.net.decoder([s_mod],
													 input_is_latent=False,
													 input_is_stylespace=True,
													 randomize_noise=False,)
							y_hat_rs = self.face_pool(y_hat)
							loss, _, _ = self.calc_loss(input_image,
				   										initial_result,
														y_hat_rs,
														target_ages=target_ages,
														input_ages=input_ages)
							
							earlystopping(loss.item())
							if earlystopping.counter == 0:
								y_hat_final = y_hat
							if earlystopping.early_stop:
								break
							else:
								loss.backward()
								optimizer.step()

						toc = time.time()
						global_time.append(toc - tic)
						
						res = tensor2im(y_hat_final[0]).resize(resize_amount)
						age_out_path_results = os.path.join(multi_path_results, age_transformer.target_age)
						os.makedirs(age_out_path_results, exist_ok=True)
						if self.opts.couple_outputs:
							multi_modal_outputs.append(res)
						else:
							im_save_path = os.path.join(age_out_path_results, f'{image_name}_{multi_i}{image_ext}')
							Image.fromarray(np.array(res)).save(im_save_path)
				
					# couple outputs
					if self.opts.couple_outputs:
						res = np.array(orig_image)
						for output in multi_modal_outputs:
							res = np.concatenate([res, np.array(output)], axis=1)
						im_save_path = os.path.join(age_out_path_results, os.path.basename(im_path))
						Image.fromarray(res).save(im_save_path)

					global_i += 1
		
			stats_path = os.path.join(age_out_path_results, 'stats.txt')
			result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
			print(result_str)

			with open(stats_path, 'w') as f:
				f.write(result_str)

	def calc_loss(self, x, y, y_hat, target_ages, input_ages):
		loss_dict = {}
		id_logs = []
		loss = 0.0
		if self.opts.id_lambda > 0:
			weights = None
			if self.opts.use_weighted_id_loss:  # compute weighted id loss only on forward pass
				age_diffs = torch.abs(target_ages - input_ages)
				weights = train_utils.compute_cosine_weights(x=age_diffs)
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, weights=weights, return_loss=True)
			loss_dict[f'loss_id'] = float(loss_id)
			loss_dict[f'id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict[f'loss_l2'] = float(loss_l2)
			l2_lambda = self.opts.l2_lambda
			loss += loss_l2 * l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict[f'loss_lpips'] = float(loss_lpips)
			lpips_lambda = self.opts.lpips_lambda
			loss += loss_lpips * lpips_lambda
		if self.opts.aging_lambda > 0:
			aging_loss, id_logs = self.aging_loss(y_hat, y, target_ages, id_logs)
			loss_dict[f'loss_aging'] = float(aging_loss)
			loss += aging_loss * self.opts.aging_lambda
		loss_dict[f'loss'] = float(loss)
		return loss, loss_dict, id_logs

	def age_id_based_perturbation(self, style_vector):
		ca = pd.read_pickle(paths_config.analyzation_path['correlation_analysis'])
		ca_idx = list(ca.index)

		ca_idx_d = {}
		for ch in ca_idx:
			pos_tmp = ch
			if ch < 512*15:
				layer = ch // 512
				channel = ch % 512
			elif ch < 512*15 + 256*3:
				pos_tmp -= 512*15
				layer = 15 + pos_tmp // 256
				channel = pos_tmp % 256
			elif ch < 512*15 + 256*3 + 128*3:
				pos_tmp -= 512*15 + 256*3
				layer = 18 + pos_tmp // 128
				channel = pos_tmp % 128
			elif ch < 512*15 + 256*3 + 128*3 + 64*3:
				pos_tmp -= 512*15 + 256*3 + 128*3
				layer = 21 + pos_tmp // 64
				channel = pos_tmp % 64
			else:
				pos_tmp -= 512*15 + 256*3 + 128*3 + 64*3
				layer = 24 + pos_tmp // 32
				channel = pos_tmp % 32
			ca_idx_d[ch] = [layer, channel]

		z_rnd = np.random.randn(1, 512)
		z_rnd = torch.from_numpy(z_rnd.astype(np.float32)).clone().cuda()
		_, o_rnd = self.net.decoder([z_rnd],
									 input_is_latent=False,
									 randomize_noise=False,
									 return_s=True)

		s_copy = [l.detach().clone() for l in style_vector]
		o_mask = [torch.zeros_like(l) for l in style_vector]

		sigma = ca['COEF_AGE'] + (1 - ca['COEF_ID'])
		sigma_max = sigma.max()
		sigma_min = sigma.min()
		o_mask_trgb = (sigma - sigma_min) / (sigma_max - sigma_min)
		
		for k, v in ca_idx_d.items():
			o_mask[v[0]][0][0][v[1]][0][0] = o_mask_trgb.at[k]

		o_rnd_d = []
		for l1, l2 in zip(o_mask, o_rnd):
			o_rnd_d.append(l1 * l2)

		s_init = []
		for l1, l2 in zip(s_copy, o_rnd_d):
			s_init.append(torch.nn.Parameter((l1 + l2).data))

		return s_init

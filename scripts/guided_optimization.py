from argparse import Namespace
import os
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import interpolate

import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp
from criteria.aging_loss import AgingLoss

class guided_optimization:
	def __init__(self, opts):
		self.test_opts = opts

		# update test options with options used during training
		ckpt = torch.load(self.test_opts.checkpoint_path, map_location='cpu')
		self.opts = ckpt['opts']
		self.opts.update(vars(self.test_opts))
		self.opts = Namespace(**self.opts)

		self.net = pSp(self.opts)
		self.net.eval()
		self.net.cuda()

		self.ext_method = lambda x, y: interpolate.interp1d(x, y, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate")
		self.aging_loss = AgingLoss()


	def optimize(self):
		print(f'Loading dataset for {self.opts.dataset_type}')
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

		for target_age in self.opts.target_age.split(','):
			print(f"Running on target age: {target_age}")
			age_list = list(range(int(target_age), int(target_age)+(self.opts.age_samples_n-1)*self.opts.age_interval+1, self.opts.age_interval))
			age_transformers = [AgeTransformer(target_age=age) for age in age_list]

			out_path_results = os.path.join(self.test_opts.exp_dir, 'inference_results', target_age)
			os.makedirs(out_path_results, exist_ok=True)

			global_time = []
			global_i = 0
			beta = self.opts.tol_beta
			for input_batch in tqdm(dataloader):
				if global_i >= self.opts.n_images:
					break
				with torch.no_grad():
					for _, input_image in enumerate(input_batch):
						tic = time.time()
						SAM_ks = []
						for age_transformer in age_transformers:
							input_age_image = age_transformer(input_image.cpu()).to('cuda')
							input_cuda = input_age_image.unsqueeze(0).cuda().float()
							_, SAM_k = self.run_on_batch(input_cuda, self.net, self.opts)

							SAM_k_tensor = []
							for SAM_k_l in SAM_k:
								SAM_k_tensor.append(SAM_k_l.squeeze())
							SAM_k_tensor = torch.cat(SAM_k_tensor)

							SAM_ks.append(SAM_k_tensor.to('cpu').detach().numpy().copy())
						
						alpha = int(target_age)
						f = self.ext_method(age_list, SAM_ks)
						count = 0
						move = 1
						prev_plus = None
						while True:
							new_latent = torch.from_numpy(f(alpha).astype(np.float32)).clone().cuda()
							result = self.run_on_batch_latents(new_latent, self.net)[0]
							age = self.aging_loss.extract_ages(result)[0]

							if abs(age - int(target_age)) < beta:
								break
							elif count == self.opts.max_steps:
								break
							
							if age < (int(target_age) - beta):
								if count != 0 and not prev_plus:
									move = move / 2
									alpha += move
								else:
									alpha += move
								prev_plus = True
							elif age > (int(target_age) + beta):
								if count != 0 and prev_plus:
									move = move / 2
									alpha -= move
								else:
									alpha -= move
								prev_plus = False
							count += 1
							
						toc = time.time()
						global_time.append(toc - tic)
						
						im_path = dataset.paths[global_i]
						resize_amount = (256, 256) if self.opts.resize_outputs else (1024, 1024)
						image_name = os.path.basename(im_path)
						im_save_path = os.path.join(out_path_results, image_name)
						res = tensor2im(result[0]).resize(resize_amount)
						Image.fromarray(np.array(res)).save(im_save_path)
						
						global_i += 1

			stats_path = os.path.join(out_path_results, 'stats.txt')
			result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
			print(result_str)

			with open(stats_path, 'w') as f:
				f.write(result_str)


	def run_on_batch(inputs, net, opts):
		result_batch, result_latents = net(inputs, randomize_noise=False, resize=opts.resize_outputs, return_s=True)
		return result_batch, result_latents

	def run_on_batch_latents(codes, net):
		sspace = [512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,256,256,256,128,128,128,64,64,64,32,32]
		codes = list(torch.split(codes, sspace))
		codes_s = []
		for code in codes:
			codes_s.append(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(code,-1),-1),0),0))
		result_batch = net.decoder([codes_s], input_is_stylespace=True, randomize_noise=False)
		return result_batch

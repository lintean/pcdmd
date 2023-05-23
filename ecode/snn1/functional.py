# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:49:34 2020

@author: win10
"""

import torch
import torch.nn.functional as F
from settings import SETTINGS
#######################

class LinearIF(torch.autograd.Function):

	@staticmethod
	def forward(ctx, spike_in, ann_output, weight, device=torch.device(SETTINGS.training.device), bias=None):
		"""
		args:
			spike_in: (N, T, in_features)
			weight: (out_features, in_features)
			bias: (out_features)
		"""
		N, T, _ = spike_in.shape
		out_features = bias.shape[0]
		pot_in = spike_in.matmul(weight.t())
		spike_out = torch.zeros_like(pot_in, device=device)
		pot_aggregate = bias.repeat(N, 1) # init the membrane potential with the bias
		spike_mask = torch.zeros_like(pot_aggregate, device=device).float()


		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			pot_aggregate += pot_in[:, t, :].squeeze()
			bool_spike = torch.ge(pot_aggregate, 1.0).float()

			bool_spike *= (1 - spike_mask)

			spike_out[:, t, :] = bool_spike
			pot_aggregate -= bool_spike

			# spike_mask += bool_spike
			# spike_mask[spike_mask > 0] = 1

		spike_count_out = torch.sum(spike_out, dim=1).squeeze()


		return spike_out, spike_count_out

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""
		grad_ann_out = grad_spike_count_out.clone()

		return None, grad_ann_out, None, None, None, None

class Conv2dIF_MaxPool(torch.autograd.Function):

	@staticmethod
	def forward(ctx, spike_in, features_in, weight, device=torch.device(SETTINGS.training.device), bias=None, stride=1, padding=0, pooling=1):
		"""
		args:
			spike_in: (N, T, in_channels, iH, iW)
			features_in: (N, in_channels, iH, iW)
			weight: (out_channels, in_channels, kH, kW)
			bias: (out_channels)
		"""
		N, T, in_channels, iH, iW = spike_in.shape
		out_channels, in_channels, kH, kW = weight.shape
		pot_aggregate = F.max_pool2d(F.conv2d(torch.zeros_like(spike_in[:,0,:,:,:]), weight, bias, stride, padding), pooling) # init the membrane potential with the bias
		_, _, outH, outW = pot_aggregate.shape
		spike_out = torch.zeros(N, T, out_channels, outH, outW, device=device)
		spike_mask = torch.zeros_like(pot_aggregate,device=device).float()


		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			pot_aggregate += F.max_pool2d(F.conv2d(spike_in[:,t,:,:,:], weight, None, stride, padding), pooling)
			bool_spike = torch.ge(pot_aggregate, 1.0).float()

			bool_spike *=(1-spike_mask)

			spike_out[:,t,:,:,:] = bool_spike
			pot_aggregate -= bool_spike

			# spike_mask += bool_spike
			# spike_mask[spike_mask > 0] = 1

		spike_count_out = torch.sum(spike_out, dim=1)

		return spike_out, spike_count_out
	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		grad_spike_count_out = grad_spike_count_out.clone()

		grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding, grad_pooling = None, \
				None, None, None, None, None, None

		return grad_spike_in, grad_spike_count_out, grad_weight, grad_device, grad_bias, \
				grad_stride, grad_padding, grad_pooling

class Conv2dIF_AvgPool(torch.autograd.Function):

	@staticmethod
	def forward(ctx, spike_in, features_in, weight, device=torch.device(SETTINGS.training.device), bias=None, stride=1, padding=0, pooling=1):
		"""
		args:
			spike_in: (N, T, in_channels, iH, iW)
			features_in: (N, in_channels, iH, iW)
			weight: (out_channels, in_channels, kH, kW)
			bias: (out_channels)
		"""
		N, T, in_channels, iH, iW = spike_in.shape
		out_channels, in_channels, kH, kW = weight.shape
		pot_aggregate = F.avg_pool2d(F.conv2d(torch.zeros_like(spike_in[:,0,:,:,:]), weight, bias, stride, padding), pooling) # init the membrane potential with the bias
		_, _, outH, outW = pot_aggregate.shape
		spike_out = torch.zeros(N, T, out_channels, outH, outW, device=device)
		spike_mask = torch.zeros_like(pot_aggregate,device=device).float()


		# Iterate over simulation time steps to determine output spike trains
		for t in range(T):
			pot_aggregate += F.avg_pool2d(F.conv2d(spike_in[:,t,:,:,:], weight, None, stride, padding), pooling)
			bool_spike = torch.ge(pot_aggregate, 1.0).float()

			bool_spike *=(1-spike_mask)

			spike_out[:,t,:,:,:] = bool_spike
			pot_aggregate -= bool_spike

			# spike_mask += bool_spike
			# spike_mask[spike_mask > 0] = 1

		spike_count_out = torch.sum(spike_out, dim=1)

		return spike_out, spike_count_out
	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		grad_spike_count_out = grad_spike_count_out.clone()

		grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding, grad_pooling = None, \
				None, None, None, None, None, None

		return grad_spike_in, grad_spike_count_out, grad_weight, grad_device, grad_bias, \
				grad_stride, grad_padding, grad_pooling

######################

class ZeroExpandInput_CNN(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input_image, T, device=torch.device(SETTINGS.training.device)):
		"""
		Args:
			input_image: normalized within (0,1)
		"""
		#N, dim = input_image.shape
		#input_image_sc = input_image
		#zero_inputs = torch.zeros(N, T-1, dim).to(device)
		#input_image = input_image.unsqueeze(dim=1)
		#input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

		#return input_image_spike, input_image_sc
		# if len(input_image.shape)==4:
		# 	input_image = input_image.squeeze(1)
		# print(input_image.shape)
		# input_image = input_image.sum(1)
		batch_size, channel, spec_length, fing_width = input_image.shape
		##################################
		# input_image_tmp = (input_image-input_image.min())/(input_image.max()-input_image.min()+1e-10) # normalized to [0-1]
		# encode_window = int(T/3)-1
		# input_image_spike = torch.zeros(batch_size,T,channel,spec_length,fing_width).to(device)
		# input_sc_index =((1-input_image_tmp)*encode_window).ceil().unsqueeze(1).long()
		# input_image_spike=input_image_spike.scatter(1,input_sc_index,1).float()
		####################################
		input_image_sc = input_image
		zero_inputs = torch.zeros(batch_size, T-1, channel, spec_length, fing_width).to(device)
		input_image = input_image.unsqueeze(dim=1)
		input_image_spike = torch.cat((input_image, zero_inputs), dim=1)
		#
		return input_image_spike, input_image_sc

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		return None, None, None

class ZeroExpandInput_MLP(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input_image, T, device=torch.device(SETTINGS.training.device)):
		"""
		Args:
			input_image: normalized within (0,1)
		"""
		N, dim = input_image.shape
		input_image_sc = input_image
		zero_inputs = torch.zeros(N, T-1, dim).to(device)
		input_image = input_image.unsqueeze(dim=1)
		input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

		return input_image_spike, input_image_sc

	@staticmethod
	def backward(ctx, grad_spike_out, grad_spike_count_out):
		"""Auxiliary function only, no gradient required"""

		return None, None, None




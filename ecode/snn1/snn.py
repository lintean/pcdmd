import torch
import torch.nn as nn
import torch.nn.functional as F
from functional import LinearIF, Conv2dIF_MaxPool, Conv2dIF_AvgPool
from settings import SETTINGS

###############################################
class LinearBN1d(nn.Module):

	def __init__(self, D_in, D_out, device=torch.device(SETTINGS.training.device), bias=True):
		super(LinearBN1d, self).__init__()
		self.linearif = LinearIF.apply
		self.device = device
		self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
		self.bn1d = torch.nn.BatchNorm1d(D_out, affine=True)

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate linear layer
		output_bn = self.bn1d(self.linear(input_features_sc))
		output = F.relu(output_bn)

		# extract the weight and bias from the surrogate linear layer
		linearif_weight = self.linear.weight#.detach().to(self.device)
		linearif_bias = self.linear.bias#.detach().to(self.device)

		bnGamma = self.bn1d.weight
		bnBeta = self.bn1d.bias
		bnMean = self.bn1d.running_mean
		bnVar = self.bn1d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Linear' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0)
		biasNorm = torch.mul(linearif_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the linearIF layer to get actual output
		# spike train
		output_st, output_sc = self.linearif(input_feature_st, output, weightNorm,  \
												self.device, biasNorm)

		return output_st, output_sc

class ConvBN2d_MaxPool(nn.Module):
	"""
	W
	"""
	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, padding=0, bias=True, weight_init=2.0, pooling=1):

		super(ConvBN2d_MaxPool, self).__init__()
		self.conv2dIF = Conv2dIF_MaxPool.apply
		self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
		self.bn2d = torch.nn.BatchNorm2d(Cout,affine=True)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.pooling = pooling

	def forward(self, input_feature_st, input_features_sc):
		# weight update based on the surrogate conv2d layer
		output_bn = F.max_pool2d(self.bn2d(self.conv2d(input_features_sc)), self.pooling)
		output = F.relu(output_bn)

		# extract the weight and bias from the surrogate conv layer
		conv2d_weight = self.conv2d.weight.detach().to(self.device)
		conv2d_bias = self.conv2d.bias.detach().to(self.device)
		bnGamma = self.bn2d.weight
		bnBeta = self.bn2d.bias
		bnMean = self.bn2d.running_mean
		bnVar = self.bn2d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(conv2d_weight.permute(1, 2, 3, 0), ratio).permute(3, 0, 1, 2)
		biasNorm = torch.mul(conv2d_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		output_features_st, output_features_sc = self.conv2dIF(input_feature_st, output,\
														weightNorm, self.device, biasNorm,\
														self.stride, self.padding, self.pooling)

		return output_features_st, output_features_sc

class ConvBN2d_AvgPool(nn.Module):
	"""
	W
	"""
	def __init__(self, Cin, Cout, kernel_size, device=torch.device(SETTINGS.training.device), stride=1, padding=0, bias=True, weight_init=2.0, pooling=1):

		super(ConvBN2d_AvgPool, self).__init__()
		self.conv2dIF = Conv2dIF_AvgPool.apply
		self.conv2d = torch.nn.Conv2d(Cin, Cout, kernel_size, stride, padding, bias=bias)
		self.bn2d = torch.nn.BatchNorm2d(Cout,affine=True)
		self.device = device
		self.stride = stride
		self.padding = padding
		self.pooling = pooling

	def forward(self, input_feature_st, input_features_sc):

		# weight update based on the surrogate conv2d layer
		output_bn = F.avg_pool2d(self.bn2d(self.conv2d(input_features_sc)), self.pooling)
		output = F.relu(output_bn)
		#output = torch.clamp(output_bn, min=0, max=T)

		# extract the weight and bias from the surrogate conv layer
		conv2d_weight = self.conv2d.weight.detach().to(self.device)
		conv2d_bias = self.conv2d.bias.detach().to(self.device)
		bnGamma = self.bn2d.weight
		bnBeta = self.bn2d.bias
		bnMean = self.bn2d.running_mean
		bnVar = self.bn2d.running_var

		# re-parameterization by integrating the beta and gamma factors
		# into the 'Conv' layer weights
		ratio = torch.div(bnGamma, torch.sqrt(bnVar))
		weightNorm = torch.mul(conv2d_weight.permute(1,2,3,0), ratio).permute(3,0,1,2)
		biasNorm = torch.mul(conv2d_bias-bnMean, ratio) + bnBeta

		# propagate the input spike train through the IF layer to get actual output
		# spike train
		output_features_st, output_features_sc = self.conv2dIF(input_feature_st, output,\
														weightNorm, self.device, biasNorm,\
														self.stride, self.padding, self.pooling)

		return output_features_st, output_features_sc
##################################################


class sDropout(nn.Module):
	def __init__(self, layerType, pDrop):
		super(sDropout, self).__init__()

		self.pKeep = 1 - pDrop
		self.type = layerType # 1: Linear 2: Conv

	def forward(self, x_st, x_sc):
		if self.training:
			T = x_st.shape[1]
			mask = torch.bernoulli(x_sc.data.new(x_sc.data.size()).fill_(self.pKeep))/self.pKeep
			x_sc_out = x_sc * mask
			x_st_out = torch.zeros_like(x_st)
			
			for t in range(T):
				# Linear Layer
				if self.type == 1:
					x_st_out[:,t,:] = x_st[:,t,:] * mask
				# Conv1D Layer
				elif self.type == 2:
					x_st_out[:,t,:,:] = x_st[:,t,:,:] * mask
				# Conv2D Layer					
				elif self.type == 3:
					x_st_out[:,t,:,:,:] = x_st[:,t,:,:,:] * mask
		else:					
			x_sc_out = x_sc
			x_st_out = x_st
			
		return x_st_out, x_sc_out
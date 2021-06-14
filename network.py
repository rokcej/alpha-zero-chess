import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		self.conv_block = ConvBlock()
		self.res_blocks = [ ResBlock() for i in range(19) ]
		self.out_block = OutBlock()
	
	def forward(self, s):
		out = self.conv_block(s)
		for i in range(len(self.res_blocks)):
			out = self.res_blocks[i](out)
		p, v = self.out_block(out)

		return p, v


class ConvBlock(nn.Module):
	def __init__(self):
		super(ConvBlock, self).__init__()

		self.conv = nn.Conv2d(119, 256, 3, stride=1, padding=1, bias=False)
		self.bn = nn.BatchNorm2d(256)

	def forward(self, s):
		out = self.conv(s)
		out = self.bn(out)
		out = F.relu(out)

		return out

class ResBlock(nn.Module):
	def __init__(self):
		super(ResBlock, self).__init__()

		self.conv1 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(256)

		self.conv2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(256)

	def forward(self, s):
		out = self.conv1(s)
		out = self.bn1(out)
		out = F.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		
		out += s
		out = F.relu(out)

		return out

class OutBlock(nn.Module):
	def __init__(self):
		super(OutBlock, self).__init__()

		# Policy head
		self.conv1_p = nn.Conv2d(256, 256, 1, stride=1)
		self.bn_p = nn.BatchNorm2d(256)
		self.conv2_p = nn.Conv2d(256, 73, 1, stride=1)

		# Value head
		self.conv_v = nn.Conv2d(256, 1, 1, stride=1)
		self.bn_v = nn.BatchNorm2d(1)
		self.fc1_v = nn.Linear(8 * 8 * 1, 256)
		self.fc2_v = nn.Linear(256, 1)

	def forward(self, s):
		# Policy head
		p = self.conv1_p(s)
		p = self.bn_p(p)
		p = F.relu(p)
		p = self.conv2_p(p)

		# Value head
		v = self.conv_v(s)
		v = self.bn_v(v)
		v = F.relu(v)
		v = torch.flatten(v, start_dim=1)
		v = self.fc1_v(v)
		v = F.relu(v)
		v = self.fc2_v(v)
		v = F.tanh(v)

		return p, v

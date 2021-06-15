import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SelfPlayDataset(Dataset):
	def __init__(self, data):
		data = np.array(data)
		self.s = data[:, 0]
		self.pi = data[:, 1]
		self.z = data[:, 2]

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.s[idx], self.pi[idx], self.z[idx]

class AlphaZeroLoss(nn.Module):
	def __init__(self):
		super(AlphaZeroLoss, self).__init__()

	def forward(p, pi, v, z):
		loss_v = ((z - v) ** 2)
		loss_p = torch.sum(-pi * p, 1)

		return torch.mean(loss_v.view(-1) + loss_p)

def train(net, train_data):
	train_set = SelfPlayDataset(train_data)
	train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
	
	criterion = AlphaZeroLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.2)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 500], gamma=0.1)

	for epoch in range(20):
		print(f"Epoch {epoch+1}: ", end="")
		for i, data in enumerate(train_loader, 0):
			s, pi, z = data
			s = s.cuda()
			pi = pi.cuda()
			z = z.cuda()

			optimizer.zero_grad()

			p, v = net(s)
			loss = criterion(p, pi, v, z)
			loss.backward()

			optimizer.step()

			print(loss.data.cpu().detach().numpy())

		scheduler.step()


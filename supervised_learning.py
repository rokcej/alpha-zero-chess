from main import BATCH_SIZE, NUM_EPOCHS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import chess
import pickle
import matplotlib.pyplot as plt
import random
import os

from network import AlphaZeroNet
import encoder_decoder as endec

SAVE_DIR = "data/supervised/"

NUM_EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 0.2

class SupervisedDataset(Dataset):
	def __init__(self, data):
		self.data = data # List of lists [fen, a, z_rel]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		s = torch.from_numpy(endec.encode_board(chess.Board(self.data[idx][0]))).float()
		pi = torch.from_numpy(np.zeros(8 * 8 * 73, dtype=np.float))
		pi[self.data[idx][1]] = 1.0
		z = self.data[idx][2]

		return s, pi, z

class AlphaZeroLoss(nn.Module):
	def __init__(self):
		super(AlphaZeroLoss, self).__init__()

	def forward(self, p, pi, v, z):
		loss_v = ((z - v) ** 2)
		loss_p = torch.sum(-pi * p, 1)

		return torch.mean(loss_v.view(-1) + loss_p)

def train(net, train_data, num_epochs, epoch_start, batch_size, learning_rate, avg_losses):
	train_set = SupervisedDataset(train_data)
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
	
	criterion = AlphaZeroLoss()
	# optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=1e-6)
	optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

	for epoch in range(epoch_start, epoch_start + num_epochs):
		avg_loss = 0.0
		with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as prog_bar:
			for i, data in enumerate(train_loader, 0):
				s, pi, z = data
				s = s.cuda()
				pi = pi.cuda()
				z = z.cuda()

				optimizer.zero_grad()

				p, v = net(s)
				loss = criterion(p, pi, v, z)
				loss.backward()
				avg_loss += loss.item()

				optimizer.step()
				
				prog_bar.update(1)
				prog_bar.set_postfix_str(f"Loss = {loss.item()}")
			
			avg_loss /= len(train_loader)
			avg_losses.append(avg_loss)
			prog_bar.set_postfix_str(f"Avg Loss = {avg_loss}")

		
		if ((epoch + 1) % 5) == 0:
			save_checkpoint = { 
				"state_dict": net.state_dict(),
				"epoch": epoch + 1,
				"avg_losses": avg_losses
			}
			torch.save(save_checkpoint, os.path.join(SAVE_DIR, f"model_{epoch+1}.pt"))
			
	save_checkpoint = { 
		"state_dict": net.state_dict(),
		"epoch": epoch + 1
	}
	torch.save(save_checkpoint, os.path.join(SAVE_DIR, f"model.pt"))
		
	return avg_losses


if __name__ == "__main__":
	train_data = []
	with open(os.path.join(SAVE_DIR, "train_data.pckl"), "rb") as f:
		train_data = pickle.load(f)

	net = AlphaZeroNet()
	net.cuda()


	epoch_start = 0
	avg_losses = []
	save_file = os.path.join(SAVE_DIR + "model.pt")
	if os.path.isfile(save_file):
		print("Loading old network...")
		load_checkpoint = torch.load(save_file)
		net.load_state_dict(load_checkpoint["state_dict"])
		epoch_start = load_checkpoint["epoch"]
		avg_losses = load_checkpoint["avg_losses"]
	else:
		if not os.path.exists(SAVE_DIR):
			os.mkdir(SAVE_DIR)
		print("Initializing new network...")
		net.initialize_parameters()
	
	random.shuffle(train_data)
	max = int(len(train_data))

	net.train()
	avg_losses = train(net, train_data[:max], NUM_EPOCHS, epoch_start, BATCH_SIZE, LEARNING_RATE, avg_losses)

	plt.plot(avg_losses)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(SAVE_DIR, f"loss_{epoch_start}.png", dpi=150))

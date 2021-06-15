from network import Network

import torch.optim as optim

def train():
	net = Network
	net.cuda()
	
	criterion = None
	optimizer = optim.Adam(net.parameters(), lr=0.2)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 500], gamma=0.1)

	for epoch in range(20):
		# Train

		# Validate

		scheduler.step()


def self_play(net):
	game = Game()
	train_data = []
	player = 1

	while not game.terminal():
		pi = mcts(game)
		train_data.append([game, player, pi, None])

	z = game.outcome(player)
	for i in range(len(train_data)):
		train_data[i][3] = z if train_data[i][1] == player else -z
	
	return train_data



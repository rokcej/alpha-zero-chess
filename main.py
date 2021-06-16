from network import Network
from self_play import self_play
from train import train


if __name__ == "__main__":
	net = Network()
	net.cuda()

	net.initialize_parameters()

	net.eval()
	train_data = self_play(net)

	# net.train()
	# train(net, train_data)


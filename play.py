from gui import GUI
from game import Game
from network import AlphaZeroNet
import encoder_decoder as endec

import time
import torch

def play(net):
	game = Game()
	gui = GUI(game.board)

	gui.draw()
	
	while not game.is_over():
		if False: # game.to_play() == 1: # Player
			pass
		else: # AI
			s = game.get_tensor().unsqueeze(0) # .cuda()
			p, v = net(s)
			p = p.squeeze(0)
			v = v.squeeze(0).item()

			actions = game.get_actions()
			probs = p[actions]

			_, a = max((prob, action) for prob, action in zip(probs, actions))
			game.apply(a)
			time.sleep(1)

		gui.draw()
		gui.handle_events()

	print(f"Outcome: {game.outcome()}")

	while gui.running:
		gui.draw()
		gui.handle_events()


if __name__ == "__main__":
	net = AlphaZeroNet()
	net.initialize_parameters()
	# net.cuda()
	net.load_state_dict(torch.load("data/models/model.pt")["state_dict"])
	net.eval()

	with torch.no_grad():
		play(net)

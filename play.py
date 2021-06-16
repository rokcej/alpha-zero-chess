from gui import GUI
from game import Game
from network import AlphaZeroNet
import encoder_decoder as endec

import chess
import time
import torch

def play_move_player(game: Game, gui: GUI):
	gui.clicked = None

	while True:
		if gui.clicked is not None:
			f, r = gui.clicked
			if gui.selected is None:
				gui.selected = gui.clicked
				for move in game.board.legal_moves:
					if move.from_square == chess.square(f, r):
						f2 = chess.square_file(move.to_square)
						r2 = chess.square_rank(move.to_square)
						gui.highlighted.add((f2, r2))
			else:
				if (f, r) in gui.highlighted:
					f0, r0 = gui.selected
					move = chess.Move(chess.square(f0, r0), chess.square(f, r))
					a = endec.encode_action(move)
					game.apply(a)
					break
				else:
					gui.selected = None
					gui.highlighted.clear()

		gui.draw()
		gui.handle_events()
	
	gui.selected = None
	gui.highlighted.clear()


def play_move_ai(game: Game, net: AlphaZeroNet):
	s = game.get_tensor().unsqueeze(0) # .cuda()
	p, v = net(s)
	p = p.squeeze(0)
	v = v.squeeze(0).item()

	actions = game.get_actions()
	probs = p[actions]

	_, a = max((prob, action) for prob, action in zip(probs, actions))
	game.apply(a)
	
	time.sleep(0.5)


def play(net: AlphaZeroNet):
	game = Game()
	gui = GUI(game.board)

	gui.draw()
	
	while not game.is_over():
		if game.to_play() == 1: # Player
			#play_move_player(game, gui)
			play_move_ai(game, net)
		else: # AI
			play_move_ai(game, net)

		gui.draw()
		gui.handle_events()

	print(f"Outcome: {game.outcome()}")

	while gui.running:
		gui.draw()
		gui.handle_events()


if __name__ == "__main__":
	net = AlphaZeroNet()
	# net.cuda()
	
	# net.initialize_parameters()
	net.load_state_dict(torch.load("data/models/model.pt")["state_dict"])

	net.eval()

	with torch.no_grad():
		play(net)

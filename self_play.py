from game import Game
from mcts import mcts

def self_play(net):
	self_play_data = []
	for iteration in range(1):
		game = Game()
		game_data = []

		while not game.is_over() and game.num_moves() < 500:
			pi, action = mcts(net, game)

			game_data.append([game.get_tensor(), pi, game.to_play()])
			game.apply(action)

		z = 0.0
		if game.is_over():
			z = game.outcome()

		for i in range(len(game_data)):
			game_data[i][2] *= z
		
		self_play_data.extend(game_data)
	
	return self_play_data


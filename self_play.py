from game import Game
from mcts import mcts

def self_play(net):
	self_play_data = []
	for iteration in range(2):
		# print(f"Game {iteration+1}:", end="")
		game = Game()
		game_data = []

		root = None

		while not game.is_over() and game.num_moves() < 500:
			# print(f" {game.num_moves()}", end="")
			pi, action, root = mcts(net, game, root)

			game_data.append([game.get_tensor(), pi, game.to_play()])
			game.apply(action)
			root = root.children[action]

		# print(f" {game.num_moves()}")

		z = 0.0
		if game.is_over():
			z = game.outcome()
		# 	print(f"Outcome: {z}")
		# else:
		# 	print("Outcomes: too many moves")

		for i in range(len(game_data)):
			game_data[i][2] *= z
		
		self_play_data.extend(game_data)
	
	return self_play_data


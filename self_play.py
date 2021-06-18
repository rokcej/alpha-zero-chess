from game import Game
from mcts import mcts
from tqdm import tqdm


def self_play(net, num_games, max_moves, num_simulations):
	self_play_data = []
	with tqdm(total=num_games, desc="Self play", unit="game") as prog_bar:
		for i_game in range(num_games):
			game = Game()
			game_data = []

			root = None

			while not game.is_over() and game.num_moves() < max_moves:
				prog_bar.set_postfix_str(f"Move {game.num_moves() + 1}")
				
				pi, action, root = mcts(net, game, num_simulations, root)

				game_data.append([game.get_tensor(), pi, game.to_play()])
				game.apply(action)
				root = root.children[action]

			z = 0.0
			if game.is_over():
				z = game.outcome()
				prog_bar.set_postfix_str(f"Outcome = {z} after {game.num_moves()} moves")
			else:
				prog_bar.set_postfix_str("Outcome = max moves reached")

			for i in range(len(game_data)):
				game_data[i][2] *= z
			
			self_play_data.extend(game_data)

			prog_bar.update(1)
	
	return self_play_data


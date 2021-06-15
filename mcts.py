import math
import numpy as np

class Node():
	def __init__(self, P: float, to_play: int):
		self.P = P
		self.N = 0
		self.W = 0.0
		self.Q = 0.0

		self.parent = None
		self.children = {}
		self.to_play = to_play

# Upper confidence bound
def ucb(parent, child):
	c_base = 19652
	c_init = 1.25
	C = math.log((1 + parent.N + c_base) / c_base) + c_init
	U = C * child.P * math.sqrt(parent.N) / (1 + child.N)

	return child.Q + U

# Monte Carlo tree search
def mcts(net, game):
	root = Node(0, game.to_play)
	expand(root, game, net)

	for simulation in range(500):
		node = root
		path = [node]
		game_sim = game.clone()

		while len(node.children) > 0:
			# Select action
			action = None
			max_score = -math.inf
			for _action, _child in node.children.items():
				score = ucb(node, _child)
				if score > max_score:
					max_score = score
					action = _action

			node = node.children[action]
			path.append(node)
			game_sim.apply(action)
		
		value = expand(node, game_sim, net)

		for _node in path:
			_node.N += 1
			_node.W += value * _node.to_play
			_node.Q = _node.W / _node.N



def expand(node, game, net):
	if game.is_over():
		return game.outcome()

	p, v = net.evaluate(game)
	a = np.random.choice(len(p), p=p)

	return value
	

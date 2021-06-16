from game import Game
import math
import numpy as np

class Node():
	def __init__(self, P: float, to_play: int):
		self.P = P
		self.N = 0
		self.W = 0.0
		self.Q = 0.0

		# self.parent = None # MEMORY LEAK???
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
def mcts(net, game, root=None):
	if root is None:
		root = Node(0, game.to_play())

	if len(root.children) == 0:
		expand(root, game, net)
		# Dirichlet noise
		actions = root.children.keys()
		noise = np.random.gamma(0.3, 1, len(actions))
		frac = 0.25
		for a, n in zip(actions, noise):
			root.children[a].P = root.children[a].P * (1.0 - frac) + n * frac

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

	# Get policy
	temp = 1
	pi = np.zeros(8 * 8 * 73)
	N_sum = sum(child.N for child in root.children.values())
	for action, child in root.children.items():
		pi[action] = (child.N ** (1.0 / temp)) / N_sum

	# Get best action
	# TODO: Softmax sample
	_, best_action = max([(c.N, a) for a, c in root.children.items()])

	return pi, best_action, root

# Expand leaf node
def expand(node: Node, game: Game, net):
	if game.is_over():
		return game.outcome()

	s = game.get_tensor().cuda().unsqueeze(0)
	p, v = net(s)
	p = p.squeeze(0)
	v = v.squeeze(0).item()

	actions = game.get_actions()
	p_sum = p[actions].sum()

	for action in actions:
		node.children[action] = Node(p[action] / p_sum, -node.to_play)

	return v
	

import chess
import numpy as np
import encoder_decoder as ED

class Game():
	def __init__(self, board=None):
		self.board = board or chess.Board()

		M, L = ED.encode_board(self.board)
		self.Ms = [ np.zeros((8, 8, 14)) for t in range(2 - 1) ]
		self.Ms.append(M)
		self.L = L

	def get_actions(self):
		return [ ED.encode_action(move) for move in self.board.legal_moves ]

	def get_tensor(self):
		return np.concatenate(self.Ms + [self.L], 2)

	def apply(self, a: int):
		self.board.push(ED.decode_action(a))

		M, L = ED.encode_board(self.board)
		self.Ms.pop(0)
		self.Ms.append(M)
		self.L = L

		return self

	def clone(self):
		return Game(self.board.copy())

	def is_over(self):
		return self.board.is_game_over()

	def outcome(self):
		outcome = self.board.outcome()
		if outcome != None:
			if outcome.winner == chess.WHITE:
				return +1
			elif outcome.winner == chess.BLACK:
				return -1
			else:
				return 0
		return None
	
	def to_play(self):
		if (len(self.board.move_stack) % 2) == 0:
			return +1 # White
		else:
			return -1 # Black



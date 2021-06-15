import chess

class Game():
	def __init__(self, board=None):
		self.board = board or chess.Board()

	def get_actions(self):
		return [ move.uci() for move in self.board.legal_moves ]

	def apply(self, a: str):
		self.board.push(chess.Move.from_uci(a))
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


def move2index(move):
	from_file = move.from_square % 8
	from_rank = move.from_square // 8

	to_file = move.to_square % 8
	to_rank = move.to_square // 8

	df = to_file - from_file
	dr = to_rank - from_rank

	plane = None

	if move.promotion != None and move.promotion != chess.QUEEN: # Underpromotion
		decode_underpromotion = {
			( 0, chess.KNIGHT): 0, ( 0, chess.BISHOP): 1, ( 0, chess.ROOK): 2,
			( 1, chess.KNIGHT): 3, ( 1, chess.BISHOP): 4, ( 1, chess.ROOK): 5,
			(-1, chess.KNIGHT): 6, (-1, chess.BISHOP): 7, (-1, chess.ROOK): 8,
		}
		plane = 64 + decode_underpromotion[(df, move.promotion)]
	elif abs(df) == 2 and abs(dr) == 1 or abs(df) == 1 and abs(dr) == 2: # Knight move
		decode_knight_move = {
			( 1,  2): 0, ( 2,  1): 1,
			( 2, -1): 2, ( 1, -2): 3,
			(-1, -2): 4, (-2, -1): 5,
			(-2,  1): 6, (-1,  2): 7,
		}
		plane = 56 + decode_knight_move[(df, dr)]
	else: # Any other move
		dist = max(abs(dr), abs(df)) - 1 # [0, 6]
		dir = None # [0, 7]
		if df == 0:
			dir = 0 if dr > 0 else 4
		elif dr == 0:
			dir = 2 if df > 0 else 6
		else:
			if df > 0:
				dir = 1 if dr > 0 else 3
			else:
				dir = 7 if dr > 0 else 5
		plane = 7 * dir + dist

	return (from_file, from_rank, plane)

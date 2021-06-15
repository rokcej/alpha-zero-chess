import chess
import numpy as np

OUT_SHAPE = (8, 8, 73)

# { key: val } -> { val: key }
def invert_dict(d):
	return { v: k for k, v in d.items() }

encode_underpromotion = {
	( 0, chess.KNIGHT): 0, ( 0, chess.BISHOP): 1, ( 0, chess.ROOK): 2,
	( 1, chess.KNIGHT): 3, ( 1, chess.BISHOP): 4, ( 1, chess.ROOK): 5,
	(-1, chess.KNIGHT): 6, (-1, chess.BISHOP): 7, (-1, chess.ROOK): 8
}
decode_underpromotion = invert_dict(encode_underpromotion)

encode_knight_move = {
	( 1,  2): 0, ( 2,  1): 1,
	( 2, -1): 2, ( 1, -2): 3,
	(-1, -2): 4, (-2, -1): 5,
	(-2,  1): 6, (-1,  2): 7
}
decode_knight_move = invert_dict(encode_knight_move)


def encode_action(move: chess.Move):
	from_file = move.from_square % 8
	from_rank = move.from_square // 8

	to_file = move.to_square % 8
	to_rank = move.to_square // 8

	df = to_file - from_file # Delta file
	dr = to_rank - from_rank # Delta rank

	plane = None

	if move.promotion != None and move.promotion != chess.QUEEN: # Underpromotion
		plane = 64 + encode_underpromotion[(df, move.promotion)]
	elif abs(df) == 2 and abs(dr) == 1 or abs(df) == 1 and abs(dr) == 2: # Knight move
		plane = 56 + encode_knight_move[(df, dr)]
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

	encoded = (from_rank * OUT_SHAPE[1] + from_file) * OUT_SHAPE[2] + plane
	return encoded


def decode_action(encoded: int, board: chess.Board):
	plane = encoded % OUT_SHAPE[2]
	tmp = encoded // OUT_SHAPE[2]
	from_file = tmp % OUT_SHAPE[1]
	from_rank = tmp // OUT_SHAPE[1]

	df = None # Delta file
	dr = None # Delta rank

	promotion = None

	if plane >= 64: # Underpromotion
		df, promotion = decode_underpromotion[plane - 64]
		dr = 1 if from_rank == 6 else -1
	elif plane >= 56: # Knight move
		df, dr = decode_knight_move[plane - 56]
	else: # Any other move
		dist = plane % 7
		dir = plane // 7
		if   dir == 0: df, dr =     0,  dist
		elif dir == 1: df, dr =  dist,  dist
		elif dir == 2: df, dr =  dist,     0
		elif dir == 3: df, dr =  dist, -dist
		elif dir == 4: df, dr =     0, -dist
		elif dir == 5: df, dr = -dist, -dist
		elif dir == 6: df, dr = -dist,     0
		elif dir == 7: df, dr = -dist,  dist

	to_file = from_file + df
	to_rank = from_rank + dr

	from_square = chess.square(from_file, from_rank)
	to_square = chess.square(to_file, to_rank)

	# Queen promotion
	if board.piece_type_at(from_square) == chess.PAWN:
		if to_rank == 0 or to_rank == 7:
			promotion = chess.QUEEN

	move = chess.Move(from_square, to_square, promotion)
	return move


def encode_board(board: chess.Board):
	M = np.zeros((8, 8, 14), dtype=int)
	L = np.zeros((8, 8, 7))

	# P1 and P2
	for rank in range(8):
		for file in range(8):
			piece = board.piece_at(chess.square(file, rank))
			if piece != None:
				color_off = 0 if piece.color == chess.WHITE else 1
				type_off = 0 # chess.PAWN
				if   piece.piece_type == chess.KNIGHT: type_off = 1
				elif piece.piece_type == chess.BISHOP: type_off = 2
				elif piece.piece_type == chess.ROOK:   type_off = 3
				elif piece.piece_type == chess.QUEEN:  type_off = 4
				elif piece.piece_type == chess.KING:   type_off = 5

				M[rank, file, color_off * 6 + type_off] = 1
	# Repetitions
	if board.is_repetition(1):
		M[:, :, 12] = 1
		if board.is_repetition(2):
			M[:, :, 13] = 1

	# Color
	if board.turn() == chess.BLACK: T[:, :, 0] = 1
	# Total moves
	L[:, :, 1] = len(board.move_stack)
	# P1 and P2 castling
	if board.has_kingside_castling_rights(chess.WHITE):  L[:, :, 2] = 1
	if board.has_queenside_castling_rights(chess.WHITE): L[:, :, 3] = 1
	if board.has_kingside_castling_rights(chess.BLACK):  L[:, :, 4] = 1
	if board.has_queenside_castling_rights(chess.BLACK): L[:, :, 5] = 1
	# No-progress count
	L[:, :, 6] = board.halfmove_clock

	return M, L


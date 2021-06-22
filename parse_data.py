import io
import chess
import chess.pgn
import numpy as np
import pickle
from tqdm import tqdm

import encoder_decoder as endec

# https://www.ficsgames.org/download.html
def read_fics(file):
	encoded_data = []
	with open(file, "r") as f:
		chunks = f.read().strip().split("\n\n")
		for i in tqdm(range(0, len(chunks), 2), desc=f"Parsing {file}"):
			params = chunks[i]
			pgn = chunks[i+1]

			# Skip disconnection outcomes
			if pgn.lower().find("disconnect") != -1:
				continue

			res_idx = params.find("[Result \"") + 9
			res = params[res_idx : res_idx+3]
		
			z_abs = None
			if   res == "1-0": z_abs = +1
			elif res == "0-1": z_abs = -1
			elif res == "1/2": z_abs = 0

			game = chess.pgn.read_game(io.StringIO(pgn))
			board = game.board()
			for move in game.mainline_moves():

				fen = board.fen()
				a = endec.encode_action(move, board)
				z_rel = z_abs if board.turn == chess.WHITE else -z_abs
				encoded_data.append([fen, a, z_rel])

				board.push(move)


			# game = chess.pgn.read_game(io.StringIO(pgn))
			# board = game.board()
			# for move in game.mainline_moves():

			# 	s = endec.encode_board(board)
			# 	a = endec.encode_action(move, board)
			# 	pi = np.zeros(8 * 8 * 73, dtype=np.float)
			# 	pi[a] = 1.0

			# 	z_rel = z_abs if board.turn == chess.WHITE else -z_abs
			# 	train_data.append([s, pi, z_rel])

			# 	board.push(move)

	return encoded_data


if __name__ == "__main__":
	train_data = []
	train_data.extend(read_fics("data/fics/2020.pgn"))
	# train_data.extend(read_fics("data/fics/2019.pgn"))

	print(f"Saving {len(train_data)} examples... ", end="")

	with open("data/supervised/train_data.pckl", "wb") as f:
		pickle.dump(train_data, f)

	print("Done!")


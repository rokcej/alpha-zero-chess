import pygame
import chess
import os

WIDTH = 640
HEIGHT = 640

LIGHT_SQUARE = (128, 128, 128)
DARK_SQUARE = (64, 64, 64)

SPRITE_DIR = "data/sprites/"

COLOR_NAMES = {
	chess.WHITE: "white",
	chess.BLACK: "black"
}
PIECE_NAMES = {
	chess.PAWN: "pawn",
	chess.KNIGHT: "knight",
	chess.BISHOP: "bishop",
	chess.ROOK: "rook",
	chess.QUEEN: "queen",
	chess.KING: "king"
}

class GUI():
	def __init__(self, board: chess.Board):
		self.board = board

		pygame.init()
		pygame.display.set_caption("AlphaZero Chess")

		self.screen = pygame.display.set_mode((640, 640))
		self.running = True

		self.images = {}
		for color, color_name in COLOR_NAMES.items():
			self.images[color] = {}
			for piece_type, piece_name in PIECE_NAMES.items():
				file_name = color_name + "_" + piece_name + ".png"
				image = pygame.image.load(os.path.join(SPRITE_DIR, file_name))
				image = pygame.transform.smoothscale(image, (WIDTH // 8, HEIGHT // 8))
				self.images[color][piece_type] = image

		pygame.display.set_icon(self.images[chess.WHITE][chess.PAWN])

	def handle_events(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False
		
	def draw_board(self):
		dx = WIDTH / 8
		dy = HEIGHT / 8
		for y in range(8):
			for x in range(8):
				color = LIGHT_SQUARE if (x + y) % 2 == 0 else DARK_SQUARE
				self.screen.fill(color, pygame.Rect(x * dx, y * dy, (x + 1) * dx, (y + 1) * dy))

	def draw_pieces(self):
		for r in range(8):
			for f in range(8):
				piece = self.board.piece_at(chess.square(f, r))
				if piece is not None:
					image = self.images[piece.color][piece.piece_type]
					self.screen.blit(image, self.get_rect(f, r))

	def get_rect(self, file, rank):
		dx = WIDTH / 8
		dy = HEIGHT / 8
		return pygame.Rect(file * dx, (7 - rank) * dy, (file + 1) * dx, (8 - rank) * dy)

	
	def draw(self):
		self.draw_board()
		self.draw_pieces()
		pygame.display.flip()


def draw_background(screen):
	dx = WIDTH / 8
	dy = HEIGHT / 8
	for y in range(8):
		for x in range(8):
			color = LIGHT_SQUARE if (x + y) % 2 == 0 else DARK_SQUARE
			screen.fill(color, pygame.Rect(x * dx, y * dy, (x + 1) * dx, (y + 1) * dy))

def draw(board: chess.Board):
	pass

def main():
	pygame.init()
	pygame.display.set_caption("AlphaZero Chess")

	screen = pygame.display.set_mode((640, 640))
	running = True

	
	draw_background(screen)

	img = pygame.image.load("./gui/sprites/white_king.png")
	img = pygame.transform.smoothscale(img, (WIDTH // 8, HEIGHT // 8))
	screen.blit(img, (0, 0))


	pygame.display.flip()

	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

if __name__ == "__main__":
	board = chess.Board()
	gui = GUI(board)
	while gui.running:
		gui.draw()
		gui.handle_events()


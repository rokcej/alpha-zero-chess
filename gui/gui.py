import pygame

WIDTH = 640
HEIGHT = 640

LIGHT_SQUARE = (128, 128, 128)
DARK_SQUARE = (64, 64, 64)

def draw_board(screen):
	dx = WIDTH / 8
	dy = HEIGHT / 8
	for y in range(8):
		for x in range(8):
			color = LIGHT_SQUARE if (x + y) % 2 == 0 else DARK_SQUARE
			screen.fill(color, pygame.Rect(x * dx, y * dy, (x + 1) * dx, (y + 1) * dy))



def main():
	pygame.init()
	pygame.display.set_caption("AlphaZero Chess")

	screen = pygame.display.set_mode((640, 640))
	running = True

	img = pygame.image.load("./gui/sprites/white_king.svg")
	screen.blit(img, (50, 50))
	
	draw_board(screen)

	pygame.display.flip()

	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

if __name__ == "__main__":
	main()

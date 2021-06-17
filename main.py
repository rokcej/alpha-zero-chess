from network import AlphaZeroNet
from self_play import self_play
from train import train

import os
import torch

SAVE_FILE = "data/models/model.pt"

NUM_STEPS = 10
NUM_GAMES = 1
NUM_EPOCHS = 600

if __name__ == "__main__":
	net = AlphaZeroNet()
	net.cuda()

	step_start = 0
	if os.path.isfile(SAVE_FILE):
		print("Loading old network...")
		load_checkpoint = torch.load(SAVE_FILE)
		net.load_state_dict(load_checkpoint["state_dict"])
		step_start = load_checkpoint["step"]
	else:
		print("Initializing new network...")
		net.initialize_parameters()

	for step in range(step_start, step_start + NUM_STEPS):
		print(f"Step {step + 1}")

		net.eval()
		with torch.no_grad():
			train_data = self_play(net, NUM_GAMES)

		net.train()
		train(net, train_data, NUM_EPOCHS)

		print("Saving... ", end="")
		save_checkpoint = { 
			"state_dict": net.state_dict(),
			"step": step + 1
		}
		torch.save(save_checkpoint, SAVE_FILE + ".bak") # Backup
		torch.save(save_checkpoint, SAVE_FILE)

		torch.cuda.empty_cache()

		print("Done!")


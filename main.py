from network import Network
from self_play import self_play
from train import train

import os
import torch

SAVE_FILE = "data/model.pt"

if __name__ == "__main__":
	net = Network()
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

	for step in range(step_start, step_start + 20):
		print(f"Step {step + 1}... ", end="")

		net.eval()
		train_data = self_play(net)

		net.train()
		train(net, train_data)

		print("Saving... ", end="")
		save_checkpoint = { 
			"state_dict": net.state_dict(),
			"step": step
		}
		torch.save(save_checkpoint, SAVE_FILE + ".bak") # Backup
		torch.save(save_checkpoint, SAVE_FILE)

		print("Done!")


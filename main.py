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
		print(f"Step {step + 1}")

		net.eval()
		with torch.no_grad():
			train_data = self_play(net, 2)

		net.train()
		train(net, train_data, 200)

		print("Saving... ", end="")
		save_checkpoint = { 
			"state_dict": net.state_dict(),
			"step": step + 1
		}
		torch.save(save_checkpoint, SAVE_FILE + ".bak") # Backup
		torch.save(save_checkpoint, SAVE_FILE)

		torch.cuda.empty_cache()

		print("Done!")


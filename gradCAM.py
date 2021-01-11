import torch
import numpy as np
import pandas as pd
from load_dataset import load_dataset
from tqdm import tqdm
from simple_cnn import SimpleCNN

class GradCAM:
	def __init__(self, model_path):
		self.model = SimpleCNN()
		self.model.load_state_dict(torch.load(model_path))
		self.model.eval()
		self.test_loader = load_dataset('test', batch_size=1)

	def extract_heatmap(self, img):
		self.model.zero_grad()
		pred = self.model(img)
		pred.backward()
		gradients = self.model.get_activations_gradient()
		pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
		activations = self.model.get_activations(img).detach()
		for i in range(activations.shape[1]):
			activations[:, i, :, :] *= pooled_gradients[i]
		heatmap = torch.mean(activations, dim=1).squeeze()
		heatmap = np.maximum(heatmap, 0) # ReLU
		heatmap /= torch.max(heatmap)
		return heatmap

	def iterator(self):
		dataset = tqdm(self.test_loader, total=len(self.test_loader))
		for batch, _ in dataset:
			heatmap = self.extract_heatmap(batch)
			print(heatmap.shape)
			exit()





if __name__ == '__main__':
	gCAM = GradCAM('./saved_models/simple_cnn_epoch_15.pt')
	gCAM.iterator()

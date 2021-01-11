import torch
import numpy as np
import pandas as pd
from load_dataset import load_dataset
from tqdm import tqdm

class SimpleCNN(torch.nn.Module):
	def __init__(self):
		super(SimpleCNN, self).__init__()
		
		# feature extractor
		self.features_conv = torch.nn.Sequential(
			torch.nn.BatchNorm2d(3),
			torch.nn.Conv2d(3, 6, kernel_size=5),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(6),
			torch.nn.Conv2d(6, 9, kernel_size=7),
			torch.nn.ReLU(),
			torch.nn.BatchNorm2d(9),
			torch.nn.Conv2d(9, 6, kernel_size=9),
			torch.nn.ReLU(),
			)
		
		# classifier
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(72600, 1024),
			torch.nn.ReLU(),
			torch.nn.Linear(1024, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 1),
			torch.nn.Sigmoid(),
			)
		
		# placeholder for the gradients
		self.gradients = None

	# hook for the gradients of the activations
	def activations_hook(self, grad):
		self.gradients = grad

	def forward(self, x):
		x = self.features_conv(x)
		h = x.register_hook(self.activations_hook)
		x = x.view((x.shape[0], -1))
		x = self.classifier(x)
		return x

	# method for the gradient extraction
	def get_activations_gradient(self):
		return self.gradients

	# method for the activation exctraction
	def get_activations(self, x):
		return self.features_conv(x)

if __name__ == '__main__':

	if torch.cuda.is_available():  
		dev = "cuda:0" 
	else:  
		dev = "cpu"  
	device = torch.device(dev)  

	train_loader = load_dataset()
	simpleCNN = SimpleCNN()
	simpleCNN = simpleCNN.to(device)

	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.SGD(simpleCNN.parameters(), lr=0.01)
	
	for epoch in range(15):
		dataset = tqdm(train_loader, total=len(train_loader))
		running_loss = []
		for batch, target in dataset:
			batch, target = batch.to(device), target.to(device)
			output = simpleCNN(batch)
			output = torch.flatten(output)
			loss = criterion(output, target.float())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss.append(loss.item())
			dataset.set_description(str({'epoch': epoch+1, 'loss': round(sum(running_loss)/len(running_loss), 4)}))
			dataset.refresh()
		dataset.close()
		torch.save(simpleCNN.state_dict(), './saved_models/simple_cnn_epoch_'+str(epoch+1)+'.pt')
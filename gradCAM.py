import torch
import numpy as np
import pandas as pd
from load_dataset import load_dataset
from tqdm import tqdm
from simple_cnn import SimpleCNN
import cv2
from PIL import Image
import os
from torchvision import transforms

class UnNormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		for t, m, s in zip(tensor, self.mean, self.std):
			t.mul_(s).add_(m)
		return tensor

class GradCAM:
	def __init__(self, model_path):
		self.model = SimpleCNN()
		self.model.load_state_dict(torch.load(model_path))
		self.model.eval()
		self.test_path = './dataset/test/'
		self.transform = transforms.Compose([transforms.Resize((128, 128)), 
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	def extract_heatmap(self, img):
		self.model.zero_grad()
		pred = self.model(img)
		val = pred.item()
		pred.backward()
		gradients = self.model.get_activations_gradient()
		pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
		activations = self.model.get_activations(img).detach()
		for i in range(activations.shape[1]):
			activations[:, i, :, :] *= pooled_gradients[i]
		heatmap = torch.mean(activations, dim=1).squeeze()
		heatmap = np.maximum(heatmap, 0) # ReLU
		heatmap = (heatmap - torch.min(heatmap))/(torch.max(heatmap) - torch.min(heatmap))
		return heatmap, val

	def combine(self, img, heatmap):
		heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
		heatmap = np.uint8(255 * heatmap)
		heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_OCEAN)
		superimposed_img = heatmap * 0.4 + img
		return superimposed_img, heatmap

	def single_image(self, path):
		img = Image.open(path)
		og_img = img
		img = self.transform(img)
		img = img.unsqueeze(0)
		heatmap, p = self.extract_heatmap(img)
		superimposed_img, heatmap = self.combine(np.array(og_img), heatmap.numpy())
		if p>0.5:
			p = 1
		else:
			p = 0
		return superimposed_img, heatmap, p
	
	def iterator(self):
		# cats
		files = [self.test_path+'cats/'+i for i in os.listdir(self.test_path+'cats/')]
		for c, cat in tqdm(enumerate(files), total=len(files)):	
			x, y, p = self.single_image(cat)
			cv2.imwrite('./gradCAM_results/heatmaps/cats/'+str(c)+'_label_'+str(p)+'.jpg', y)
			cv2.imwrite('./gradCAM_results/superimposed/cats/'+str(c)+'_label_'+str(p)+'.jpg', x)

		files = [self.test_path+'dogs/'+i for i in os.listdir(self.test_path+'dogs/')]
		for c, cat in tqdm(enumerate(files), total=len(files)):
			x, y, p = self.single_image(cat)
			cv2.imwrite('./gradCAM_results/heatmaps/dogs/'+str(c)+'_label_'+str(p)+'.jpg', y)
			cv2.imwrite('./gradCAM_results/superimposed/dogs/'+str(c)+'_label_'+str(p)+'.jpg', x)
		
if __name__ == '__main__':
	gCAM = GradCAM('./saved_models/simple_cnn_epoch_15.pt')
	gCAM.iterator()

import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image

class VGG(torch.nn.Module):
	def __init__(self):
		super(VGG, self).__init__()
		
		# get the pretrained VGG19 network
		self.vgg = torchvision.models.vgg19(pretrained=True)
		
		# disect the network to access its last convolutional layer
		self.features_conv = self.vgg.features[:36]
		
		# get the max pool of the features stem
		self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		
		# get the classifier of the vgg19
		self.classifier = self.vgg.classifier
		
		# placeholder for the gradients
		self.gradients = None
	
	# hook for the gradients of the activations
	def activations_hook(self, grad):
		self.gradients = grad
		
	def forward(self, x):
		x = self.features_conv(x)
		
		# register the hook
		h = x.register_hook(self.activations_hook)
		
		# apply the remaining pooling
		x = self.max_pool(x)
		x = x.view((1, -1))
		x = self.classifier(x)
		return x
	
	# method for the gradient extraction
	def get_activations_gradient(self):
		return self.gradients
	
	# method for the activation exctraction
	def get_activations(self, x):
		return self.features_conv(x)

def extract_heatmap(img, pred_val, name):
	pred = vgg(img)

	pred[:, pred_val].backward()

	gradients = vgg.get_activations_gradient()

	print(gradients.shape)

	pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

	activations = vgg.get_activations(img).detach()

	for i in range(512):
		activations[:, i, :, :] *= pooled_gradients[i]

	heatmap = torch.mean(activations, dim=1).squeeze()

	heatmap = np.maximum(heatmap, 0) # ReLU

	heatmap /= torch.max(heatmap)


	plt.matshow(heatmap.squeeze())
	plt.savefig(name+'_heatmap.jpg')
	plt.cla()
	return heatmap

def combine(name, heatmap, path):
	img = cv2.imread(path)
	heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
	heatmap = np.uint8(255 * heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	superimposed_img = heatmap * 0.4 + img
	cv2.imwrite(name+'_map.jpg', superimposed_img)

def load_img(path):
	img = Image.open(path)
	img = transform(img)
	img = img.unsqueeze(0)
	return img

if __name__ == '__main__':

	transform = transforms.Compose([transforms.Resize((224, 224)), 
									transforms.ToTensor(),
									transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	path = './vgg_images/elephant/elephant.jpeg'
	img = load_img(path)
	vgg = VGG()
	vgg.eval()
	pred = vgg(img).argmax(dim=1)
	heatmap = extract_heatmap(img, pred.item(), './vgg_results/elephant')
	combine('./vgg_results/elephant', heatmap, path=path)

	path = './vgg_images/shark/shark.jpeg'
	img = load_img(path)
	vgg = VGG()
	vgg.eval()
	pred = vgg(img).argmax(dim=1)
	heatmap = extract_heatmap(img, pred.item(), './vgg_results/shark')
	combine('./vgg_results/shark', heatmap, path=path)
import torch
from torchvision import datasets, transforms

def load_dataset(datatype='train'):
	if datatype=='train':
		path = './dataset/train/'
	else:
		path = './dataset/test/'
	transform = transforms.Compose([transforms.ToTensor()])

	dataset = datasets.ImageFolder(path, transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
	return dataloader
import torch
from torchvision import datasets, transforms

def load_dataset(datatype='train', batch_size=16):
	if datatype=='train':
		path = './dataset/train/'
	else:
		path = './dataset/test/'
	transform = transforms.Compose([transforms.Resize((128, 128)), 
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	dataset = datasets.ImageFolder(path, transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return dataloader
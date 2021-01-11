import torch
import numpy as np
import pandas as pd
from load_dataset import load_dataset
from tqdm import tqdm
from simple_cnn import SimpleCNN

if __name__ == '__main__':
	if torch.cuda.is_available():  
		dev = "cuda:0" 
	else:  
		dev = "cpu"  
	device = torch.device(dev)  

	test_loader = load_dataset('test')
	criterion = torch.nn.BCELoss()
	progress = []
	for epoch in range(15):
		PATH = './saved_models/simple_cnn_epoch_'+str(epoch+1)+'.pt'
		simpleCNN = SimpleCNN()
		simpleCNN.load_state_dict(torch.load(PATH))
		simpleCNN = simpleCNN.to(device)
		simpleCNN.eval()

		dataset = tqdm(test_loader, total=len(test_loader))
		running_loss = []
		running_acc = []
		for batch, target in dataset:
			batch, target = batch.to(device), target.to(device)
			output = simpleCNN(batch)
			output = torch.flatten(output)
			loss = criterion(output, target.float())
			running_loss.append(loss.item())
			output = torch.round(output)
			acc = torch.sum(output == target)/output.shape[0]
			running_acc.append(acc.item())

			dataset.set_description(str({'epoch': epoch+1, 'loss': round(sum(running_loss)/len(running_loss), 4), 'acc': round(sum(running_acc)/len(running_acc), 4)}))
			dataset.refresh()
		dataset.close()
		prog = [epoch+1, sum(running_loss)/len(running_loss), sum(running_acc)/len(running_acc)]
		progress.append(prog)
	progress = pd.DataFrame(np.array(progress))
	progress.columns = ['epoch', 'loss', 'acc']
	progress.to_csv('simple_cnn_test_results.csv', index=False)
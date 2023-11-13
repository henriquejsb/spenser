
import torch, torch.nn as nn
import snntorch as snn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from snntorch import functional as SF
from evotorch.neuroevolution import SupervisedNE
import matplotlib.pyplot as plt
from evotorch.neuroevolution.net import count_parameters
from time import time
from snntorch import spikegen
import math
from Common import *


num_epochs = 1


net = SNN().to(device)
print(f'Network has {count_parameters(net)} parameters')

input()

train_dataset,test_dataset = load_mnist()





optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
 


dataloading_time = 0
spikegen_time = 0
forward_time = 0
learning_time = 0
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

acc_hist = []
loss_hist = []
# training loop

start = time()

for epoch in range(num_epochs):
    for i, (_data, targets) in enumerate(iter(trainloader)):
        

        #data = _data.transpose(0,1).to(device)
        data = _data.to(device)
        
        targets = targets.to(device)

        net.train()

        
        spk_rec = net(data)
        


        loss_val = loss_fn(spk_rec, targets)
        
        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


        
        # Store loss history for future plotting
        with torch.no_grad():
            
            
            loss_hist.append(loss_val.item())
            
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)

        if True:
            print(f"Epoch {epoch}, Iteration {i}/{len(trainloader)} \nTrain Loss: {loss_val.item():.2f} Accuracy: {acc * 100:.2f}%")

   



print(f"Finished training after {time() - start} seconds.")


evaluate(net,test_dataset)
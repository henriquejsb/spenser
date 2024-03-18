
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


num_epochs = 100



timestamp = str(int(time()))

LOGGER_FILE = 'logger_adam_'+timestamp+'.txt'
NETWORK_NAME = 'adam_'+timestamp+'.checkpoint'


net = SNN().to(device)
print(f'Network has {count_parameters(net)} parameters')

input()

train_dataset,test_dataset = load_fashion()





optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
 


dataloading_time = 0
spikegen_time = 0
forward_time = 0
learning_time = 0
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

acc_hist = []
loss_hist = []
test_acc_vals = []
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
            
            acc = SF.accuracy_rate(spk_rec, targets,population_code=True, num_classes=10)
            acc_hist.append(acc)

        if True:
            print(f"Epoch {epoch}, Iteration {i}/{len(trainloader)} \nTrain Loss: {loss_val.item():.2f} Accuracy: {acc * 100:.2f}%")

        if i%10 == 0:
            test_acc = evaluate(net,test_dataset)
            test_acc_vals += [test_acc]
            aux_logger = {}
            aux_logger["test_acc"] = test_acc_vals
            aux_logger["train_acc"] = acc_hist
            aux_logger["train_loss"] = loss_hist
            with open(LOGGER_FILE,"w") as f:
                f.write(str(aux_logger))
            save(net,NETWORK_NAME)


print(f"Finished training after {time() - start} seconds.")

aux_logger["time"] = time() - start
final_test = evaluate(net,test_dataset)
aux_logger["final_test"] = final_test

with open(LOGGER_FILE,"w") as f:
    f.write(str(aux_logger))

save(net,NETWORK_NAME)
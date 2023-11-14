
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

batch_size = 32
data_path='./data/mnist'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

loss_fn = SF.mse_count_loss(correct_rate=1, incorrect_rate=0)

# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 10
beta = 0.5





# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])



def load_mnist():
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)


    # Create DataLoaders
    #train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    #test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    #return train_loader, test_loader
    return mnist_train, mnist_test

def save(model):
    # Save the models phenotype and weights 
    #opt_stade_dict = None
    #if optimizer.is_backprop:
    #    opt_stade_dict = optimizer.state_dict()
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        #"optimizer_state_dict": opt_stade_dict,
    }
    torch.save(checkpoint, 'model.checkpoint')



def evaluate(net,test_dataset):
    net.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False)
    test_loss = 0
    correct = 0
    total = 0
    correct = 0

    save(net)      
        
    with torch.no_grad():
        for data, target in test_loader:
            #print(data)
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            test_loss += loss_fn(output, target).item() * data.shape[0]
            _, pred = output.sum(dim=0).max(1)
            #print(pred)
            acc = SF.accuracy_rate(output, target)
            total += target.size(0)
            correct += (pred == target).sum().item()
            #pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / total))
    print(f"Correct/Total = {correct} / {total} ")
    print("SF Accuracy: " + str(acc))
    return correct / len(test_loader.dataset)

class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        for param in self.fc1.parameters():
            #pass
            #param.trainable = False
            param.requires_grad = False     
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        x = spikegen.rate(x, num_steps=num_steps)

        for i in range(num_steps):

            cur1 = self.fc1(self.flat(x[i]))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            #mem2_rec.append(mem2)
        #print(torch.stack(spk2_rec, dim=0))
        return torch.stack(spk2_rec, dim=0)#, torch.stack(mem2_rec, dim=0)


'''
class SNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        #self.flat = nn.Flatten()
        self.dp1 = nn.Dropout(p=0.3225)
        self.dp2 = nn.Dropout(p=0.01001)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(784,10,bias=True)
        self.lif2 = snn.Leaky(beta=beta)
        # Initialize layers
        #self.fc1 = nn.Linear(num_inputs, num_hidden)
        #elf.lif1 = snn.Leaky(beta=beta)
        #self.fc2 = nn.Linear(num_hidden, num_outputs)
        #self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        #mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []
        x = spikegen.rate(x, num_steps=num_steps)
        #print(x.size())
        for i in range(num_steps):

            #cur1 = self.fc1(self.flat(x))
            #spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc1(self.flat(self.dp2(self.dp1(x[i]))))
            #cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            #mem2_rec.append(mem2)
        #print(torch.stack(spk2_rec, dim=0))
        return torch.stack(spk2_rec, dim=0)#, torch.stack(mem2_rec, dim=0)
'''
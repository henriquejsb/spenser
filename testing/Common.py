
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
import torch.nn.functional as F
from snntorch import surrogate
from snntorch import utils
import random
import numpy as np


SEED = 10


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)




batch_size = 128





data_path='../data/mnist'
data_cifar = '../data/cifar-10'
data_fashion = '../data/fashion'


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

spike_grad = surrogate.fast_sigmoid(slope=25)
loss_fn = SF.mse_count_loss(correct_rate=1, incorrect_rate=0)

# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 5
beta = 0.5


pop_coding = 10


# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

transform_cifar =  transforms.Compose([
                transforms.Resize((32,32)),
                #transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,)),
                ])


def load_mnist():
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)


    # Create DataLoaders
    #train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    #test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    #return train_loader, test_loader
    return mnist_train, mnist_test

def load_fashion():
    mnist_train = datasets.FashionMNIST(data_fashion, train=True, download=True, transform=transform)
    mnist_test = datasets.FashionMNIST(data_fashion, train=False, download=True, transform=transform)


    # Create DataLoaders
    #train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    #test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    #return train_loader, test_loader
    return mnist_train, mnist_test


def load_cifar():
    mnist_train = datasets.CIFAR10(data_cifar, train=True, download=True, transform=transform_cifar)
    mnist_test = datasets.CIFAR10(data_cifar, train=False, download=True, transform=transform_cifar)


    # Create DataLoaders
    #train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    #test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    #return train_loader, test_loader
    return mnist_train, mnist_test




def save(model,dir):
    # Save the models phenotype and weights 
    #opt_stade_dict = None
    #if optimizer.is_backprop:
    #    opt_stade_dict = optimizer.state_dict()
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        #"optimizer_state_dict": opt_stade_dict,
    }
    torch.save(checkpoint, dir)



def evaluate(net,test_dataset):
    net.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False)
    test_loss = 0
    correct = 0
    total = 0
    correct = 0

    #save(net)      
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            #print(data)
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            test_loss += loss_fn(output, target).item() * data.shape[0]
            _, pred = output.sum(dim=0).max(1)
            #print(output.sum(dim=0))
            #print(pred)
            #print("-------------")
            #print(pred)
            acc += output.size(1) * SF.accuracy_rate(output, target)
            total += output.size(1)
            correct += (pred == target).sum().item()
            #pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100 *acc/total))
    print(f"Correct/Total = {correct} / {total} ")
    print("SF Accuracy: " + str(acc/total))
    #return correct / len(test_loader.dataset)
    return acc / total


'''

class SNN(nn.Module):
    #population coding
    def __init__(self):
        super().__init__()
        self.module_seq = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    nn.BatchNorm2d(12),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    nn.BatchNorm2d(64),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10 * pop_coding),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)
    
    def forward(self,x):
        mem_rec = []
        spk_rec = []
        
        utils.reset(self.module_seq)  # resets hidden states for all LIF neurons in net
        
        x = spikegen.rate(x, num_steps=num_steps)
        
        

        
        for i in range(num_steps):
            spk_out, mem_out = self.module_seq(x[i])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0)#, torch.stack(mem_rec)



'''







class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_seq = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    nn.BatchNorm2d(12),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    nn.BatchNorm2d(64),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)
    
    def forward(self,x):
        mem_rec = []
        spk_rec = []
        
        utils.reset(self.module_seq)  # resets hidden states for all LIF neurons in net
        
        x = spikegen.rate(x, num_steps=num_steps)
        
        

        
        for i in range(num_steps):
            spk_out, mem_out = self.module_seq(x[i])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec, dim=0)#, torch.stack(mem_rec)


'''
class SNN2(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64*4*4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk_rec = []
        
        for i in range(num_steps):
            cur1 = F.batch_norm(F.max_pool2d(self.conv1(x[i]), 2))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = F.batch_norm(F.max_pool2d(self.conv2(spk1), 2))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.view(batch_size, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk_rec.append(spk3)
        return torch.stack(spk_rec, dim=0)#, mem3

'''
'''
class SNN2(nn.Module):
    def __init__(self):
        super().__init__()
        #self.cv1 = nn.Conv2d()
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
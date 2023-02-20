import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils
from torch.utils.data import DataLoader, random_split, Subset
from snntorch import spikegen
from sklearn.model_selection import train_test_split
import numpy as np
# Training Parameters
batch_size=32
data_path='./data/mnist'
num_classes = 10  # MNIST has 10 output classes

# Temporal Dynamics
num_steps = 10

# Torch Variables
dtype = torch.float

# Define a transform
transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

def load_MNIST(train=True):
    mnist = datasets.MNIST(data_path, train=train, download=True, transform=transform)
    return mnist

def auxiliary():
    subset = 100
    mnist_train = utils.data_subset(mnist_train, subset)

    print(mnist_train.data)
    spike_data = spikegen.rate(mnist_train.data, num_steps=num_steps)
    print(spike_data)

    #train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)



def load_dataset(dataset, config):

    mnist_train = load_MNIST(train=True)
    mnist_test = load_MNIST(train=False)

    #evo_train, evo_test = random_split(mnist_train, )
    #Splitting the data
    
    subset = 10
    mnist_train = utils.data_subset(mnist_train, subset)
    
    #
    #print(res)
    # print(mnist_train[0])
    # print(len(mnist_train))
    
    
    indices = np.arange(0,len(mnist_train))

    evo_train_idx, evo_test_idx = train_test_split(indices, test_size = 0.33, shuffle=True, stratify=mnist_train.targets)
    
    # print(evo_train_idx)
    
    # print(type(evo_test_idx))
   
    evo_train = Subset(mnist_train, evo_train_idx)
    evo_test = Subset(mnist_train, evo_test_idx)
    print(f"Evo train dataset has {len(evo_train)} samples")
    print(f"Evo test dataset has {len(evo_test)} samples")
    evo_train = DataLoader(evo_train, batch_size=batch_size)
    evo_test = DataLoader(evo_test, batch_size=batch_size)
    test = DataLoader(mnist_test, batch_size=batch_size)

    dataset = {
        "evo_train": evo_train,
        "evo_test": evo_test,
        "test": test
    }
    return dataset


if __name__ == '__main__':
    load_dataset('asd',None)

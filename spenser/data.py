import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils
from torch.utils.data import DataLoader, random_split, Subset
from snntorch import spikegen
from sklearn.model_selection import train_test_split
import numpy as np

# Training Parameters
MNIST='./data/mnist'
FMNIST='./data/fashion_mnist'
CIFAR_10='./data/cifar-10'


# Torch Variables
dtype = torch.float

# Define a transform
transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

transform_cifar =  transforms.Compose([
                transforms.Resize((32,32)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

def load_MNIST(train=True):
    mnist = datasets.MNIST(MNIST, train=train, download=True, transform=transform)
    return mnist

def load_FashionMNIST(train=True):
    fmnist = datasets.FashionMNIST(FMNIST, train=train, download=True, transform=transform)
    return fmnist

def load_CIFAR10(train=True):
    cifar_10 = datasets.CIFAR10(CIFAR_10, train=train, download=True, transform=transform_cifar)
    return cifar_10

def prepare_dataset(trainset,testset,subset,batch_size,num_steps):


    train = utils.data_subset(trainset, subset)

    indices = np.arange(0,len(train))

    evo_train_idx, evo_test_idx = train_test_split(indices, test_size = 0.3, shuffle=True, stratify=train.targets)
    
    evo_train = Subset(train, evo_train_idx)
    evo_test = Subset(train, evo_test_idx)

    print(f"Evo train dataset has {len(evo_train)} samples")
    print(f"Evo test dataset has {len(evo_test)} samples")
 
    evo_train = DataLoader(evo_train,batch_size=batch_size,shuffle=True,num_workers=8)
    evo_test = DataLoader(evo_test,batch_size=batch_size,shuffle=True,num_workers=8)
    test = DataLoader(testset,batch_size=batch_size,shuffle=True)

    dataset = {
        "evo_train": evo_train,
        "evo_test": evo_test,
        "test": test
    }
    return dataset


def load_dataset(dataset, config):
    subset = int(config["TRAINING"]["subset"])
    batch_size = int(config["TRAINING"]["batch_size"])
    num_steps = int(config["TRAINING"]["num_steps"])

    if dataset == 'mnist':
        trainset = load_MNIST(train=True)
        testset = load_MNIST(train=False)
        input_size = (1,28,28)
    elif dataset == 'fashion_mnist':
        trainset = load_FashionMNIST(train=True)
        testset = load_FashionMNIST(train=False)
        input_size = (1,28,28)
    elif dataset == 'cifar-10':
        trainset = load_CIFAR10(train=True)
        testset = load_CIFAR10(train=False)
        input_size = (1,32,32)
    else:
        print("Error: the dataset is not valid")
        exit(-1)

    dataset = prepare_dataset(trainset,testset,subset,batch_size,num_steps)
    dataset["input_size"] = input_size
    return dataset



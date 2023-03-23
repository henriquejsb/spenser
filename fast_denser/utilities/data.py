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

    CONVERT = False
    train = utils.data_subset(trainset, subset)

    indices = np.arange(0,len(train))

    evo_train_idx, evo_test_idx = train_test_split(indices, test_size = 0.3, shuffle=True, stratify=train.targets)
    

    evo_train = Subset(train, evo_train_idx)
    evo_test = Subset(train, evo_test_idx)

    print(f"Evo train dataset has {len(evo_train)} samples")
    print(f"Evo test dataset has {len(evo_test)} samples")

    if CONVERT:
        evo_train = DataLoader(evo_train)
        evo_test = DataLoader(evo_test)
        test = DataLoader(testset)

        #evo_train = preprocess_dataloader(evo_train, batch_size, num_steps)
        #evo_test = preprocess_dataloader(evo_test,batch_size, num_steps)
        #test = preprocess_dataloader(test,batch_size,num_steps)

    else:
        evo_train = DataLoader(evo_train,batch_size=batch_size,shuffle=True,num_workers=8,persistent_workers=True)
        evo_test = DataLoader(evo_test,batch_size=batch_size,shuffle=True,num_workers=8,persistent_workers=True)
        test = DataLoader(testset,batch_size=batch_size,shuffle=True)

    dataset = {
        "evo_train": evo_train,
        "evo_test": evo_test,
        "test": test
    }
    return dataset

def preprocess_dataloader(data_loader, batch_size, num_steps):
    instances = []
    i = 0
    for x, y in data_loader:
        i += 1
        #print("X",x.shape)
        #print("X.data",x.data.shape)
        x = spikegen.rate(x[0].data, num_steps=num_steps)
        #print("Spike X",x.shape)
        instances.append((x, y))
    print("Iterations:",i)
    new_data_loader = DataLoader(instances, batch_size=batch_size, pin_memory=False, num_workers=0)
    return new_data_loader




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
        input_size = (3,32,32)
    else:
        print("Error: the dataset is not valid")
        exit(-1)

    dataset = prepare_dataset(trainset,testset,subset,batch_size,num_steps)
    dataset["input_size"] = input_size
    return dataset


def test_load_dataset(dataset):
    subset = 1
    batch_size = 32
    num_steps = 10

    if dataset == 'mnist':
        trainset = load_MNIST(train=True)
        testset = load_MNIST(train=False)
    elif dataset == 'fashion_mnist':
        trainset = load_MNIST(train=True)
        testset = load_MNIST(train=False)
    else:
        print("Error: the dataset is not valid")
        exit(-1)

    #dataset = prepare_dataset(trainset,testset,subset,batch_size)
    #dataset = prepare_dataset_2(trainset,testset,subset,batch_size, num_steps)
    return dataset

if __name__ == '__main__':
    test_load_dataset('mnist')

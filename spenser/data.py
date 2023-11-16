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

class ImgToSpikeTrain(object):

    def __init__(self,time_steps):
        self.time_steps = time_steps
    
    def __call__(self,sample):
        #print("SAMPLE",sample)
        return spikegen.rate(sample, num_steps=self.time_steps)

class RepeatImage(object):

    def __init__(self,time_steps):
        self.time_steps = time_steps
    
    def __call__(self,sample):
        #print("SAMPLE",sample)
        #return spikegen.rate(sample, num_steps=self.time_steps)
        return np.broadcast_to(sample,(self.time_steps,)+sample.shape)





def load_MNIST(train=True,transform=None):
    mnist = datasets.MNIST(MNIST, train=train, download=True, transform=transform)
    return mnist

def load_FashionMNIST(train=True,transform=None):
    fmnist = datasets.FashionMNIST(FMNIST, train=train, download=True, transform=transform)
    return fmnist

def load_CIFAR10(train=True,transform=None):
    cifar_10 = datasets.CIFAR10(CIFAR_10, train=train, download=True, transform=transform)
    return cifar_10

def prepare_dataset(trainset,testset,subset,batch_size,num_steps):


    train = utils.data_subset(trainset, subset)

    indices = np.arange(0,len(train))

    evo_train_idx, evo_test_idx = train_test_split(indices, test_size = 0.3, shuffle=True, stratify=train.targets)
    
    evo_train_original = Subset(train, evo_train_idx)
    evo_test_original = Subset(train, evo_test_idx)

    print(f"Evo train dataset has {len(evo_train_original)} samples")
    print(f"Evo test dataset has {len(evo_test_original)} samples")
 
    evo_train = DataLoader(evo_train_original,batch_size=batch_size,shuffle=True,num_workers=0)
    evo_test = DataLoader(evo_test_original,batch_size=batch_size,shuffle=True,num_workers=0)
    test = DataLoader(testset,batch_size=batch_size,shuffle=True)

    dataset = {
        "evo_train": evo_train,
        "evo_test": evo_test,
        "test": test,
        "evo_train_original": evo_train_original,
        "evo_test_original": evo_test_original,
        "test_original": testset
    }
    return dataset


def load_dataset(dataset, config):
    subset = int(config["TRAINING"]["subset"])
    batch_size = int(config["TRAINING"]["batch_size"])
    num_steps = int(config["TRAINING"]["num_steps"])
    # Define a transform
    transform = transforms.Compose([
                    transforms.Resize((28,28)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,)),
                    ImgToSpikeTrain(num_steps)])

    transform_cifar =  transforms.Compose([
                    transforms.Resize((32,32)),
                    #transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,)),
                    ImgToSpikeTrain(num_steps)])

    transform_cifar_original =  transforms.Compose([
                    transforms.Resize((32,32)),
                    #transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,)),
                    RepeatImage(num_steps)])

    if dataset == 'mnist':
        trainset = load_MNIST(train=True,transform=transform)
        testset = load_MNIST(train=False,transform=transform)
        input_size = (1,28,28)
    elif dataset == 'fashion_mnist':
        trainset = load_FashionMNIST(train=True,transform=transform)
        testset = load_FashionMNIST(train=False,transform=transform)
        input_size = (1,28,28)
    elif dataset == 'cifar-10':
        trainset = load_CIFAR10(train=True,transform=transform_cifar)
        testset = load_CIFAR10(train=False,transform=transform_cifar)
        input_size = (3,32,32)
    elif dataset == 'cifar-10-original':
        trainset = load_CIFAR10(train=True,transform=transform_cifar_original)
        testset = load_CIFAR10(train=False,transform=transform_cifar_original)
        input_size = (3,32,32)
    else:
        print("Error: the dataset is not valid")
        exit(-1)

    dataset = prepare_dataset(trainset,testset,subset,batch_size,num_steps)
    dataset["input_size"] = input_size
    return dataset



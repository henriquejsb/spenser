from bindsnet.datasets import MNIST, CIFAR10
from bindsnet.encoding import PoissonEncoder, poisson
import os
from torchvision import transforms


def load_dataset(dataset, config):
    time = config["TRAINING"]["time"]
    dt = config["TRAINING"]["dt"]
    intensity = 128
    train = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join("..", "..", "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )
    test = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join("..", "..", "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
        train=False
    )
    dataset = {"train":train,"test":test}
    return dataset


def test_load_dataset():
    # Load MNIST data.
    time = 100
    dt = 1
    intensity = 64
    dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join("..", "..", "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
        ),
    )
    print(type(dataset))

if __name__ == '__main__':
    test_load_dataset()
import tonic
import tonic.transforms as transforms
from tonic import CachedDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import random



def load_dataset(dataset, config):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=5000)
                                     ])
    

    trainset = np.asarray(tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True))
    testset = np.asarray(tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False))

    evo_train, evo_val = train_test_split(trainset,
                                            test_size=0.2,
                                            shuffle=True)
    

    #Split the trainset 

    
    
    
    
    
    
    
    #cached_trainset = CachedDataset(trainset, cache_path='./cache/nmnist/train')
    return trainset


def test_load_dataset():
    #dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
    #events, target = dataset[0]
    #print(events)
    #tonic.utils.plot_event_grid(events)

    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=5000)
                                     ])
    

    trainset = tonic.datasets.NMNIST(save_to='./data', 
                                    transform=frame_transform, 
                                    train=True)
                        
    print(len(trainset))
    print(type(trainset))
    trainset = np.asarray(trainset)
    print("oi")
    short_trainset = np.random.choice(trainset,500)
    #short_trainset = trainset[500]
    print(len(short_trainset))
    print(type(short_trainset))
    
    #evo_train, evo_val = train_test_split(short_trainset,
    #                                        test_size=0.2,
    #                                        shuffle=True)
    
    #tonic.utils.plot_event_grid(events)
    return trainset

if __name__ == '__main__':
    test_load_dataset()
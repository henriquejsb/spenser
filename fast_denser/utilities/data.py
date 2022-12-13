import tonic
import tonic.transforms as transforms
from tonic import CachedDataset
from torch.utils.data import DataLoader

def load_dataset(dataset, config):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=5000)
                                     ])
    

    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    #cached_trainset = CachedDataset(trainset, cache_path='./cache/nmnist/train')
    return trainset


def test_load_dataset():
    #dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
    #vents, target = dataset[0]
    #print(events)
    #tonic.utils.plot_event_grid(events)

    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=5000)
                                     ])
    

    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    return trainset

if __name__ == '__main__':
    test_load_dataset()
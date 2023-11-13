
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

WRITE = False


network = SNN().to(device)
print(f'Network has {count_parameters(network)} parameters')
print(f'Popsize should be {4 + int(3 * math.log(count_parameters(network)) )}')
input()


train_dataset,test_dataset = load_mnist()

NETWORK = SNN()


mnist_problem = SupervisedNE(
    train_dataset,  # Using the dataset specified earlier
    NETWORK,  # Training the SNN module designed earlier
    #SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0),  # Minimizing CrossEntropyLoss
    SF.ce_count_loss(),  # Minimizing CrossEntropyLoss
    minibatch_size = 32,  # With a minibatch size of 32
    num_minibatches = 1,
    common_minibatch = True,  # Always using the same minibatch across all solutions on an actor
    num_actors = 1,  # The total number of CPUs used
    num_gpus_per_actor = 'max',  # Dividing all available GPUs between the 4 actors
    device=device
    #subbatch_size = 50,  # Evaluating solutions in sub-batches of size 50 ensures we won't run out of GPU memory for individual workers
)

from evotorch.algorithms import PGPE,CMAES



searcher = CMAES(mnist_problem,stdev_init=.1,separable=True  )




from evotorch.logging import StdOutLogger, PandasLogger
stdout_logger = StdOutLogger(searcher, interval = 1)
pandas_logger = PandasLogger(searcher, interval = 1)



acc_vals = []

start = time()

for _ in range(1000):
    searcher.run(10)

    final_sol = torch.squeeze(searcher.status["center"])
    
    print("SOL SIZE: ",final_sol.size())
    input()
    net = mnist_problem.parameterize_net(final_sol).to(device)

    #loss = torch.nn.CrossEntropyLoss()
    acc = evaluate(net,test_dataset)
    acc_vals += [ acc]

    print(acc_vals)

    metrics = {}
    metrics['cmaes_logger'] = pandas_logger.to_dataframe()
    aux_logger = {}
    #aux_logger["iter"] = metrics['cmaes_logger']['iter'].tolist()
    aux_logger["stepsize"] = metrics['cmaes_logger']['stepsize'].tolist()
    aux_logger["mean_eval"] = metrics['cmaes_logger']["mean_eval"].tolist()
    aux_logger["median_eval"] = metrics['cmaes_logger']["median_eval"].tolist()
    aux_logger["pop_best_eval"] = metrics['cmaes_logger']["pop_best_eval"].tolist()
    aux_logger['accuracy'] = acc_vals
    #metrics['cmaes_logger'] = aux_logger
    if WRITE:
        with open('logger.txt','w') as f:
            f.write(str(aux_logger))

finish = time() - start
aux_logger['time'] = finish

if WRITE:
    with open('logger.txt','w') as f:
        f.write(str(aux_logger))
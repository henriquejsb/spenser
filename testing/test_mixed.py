
import torch, torch.nn as nn
import snntorch as snn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from snntorch import functional as SF
from evotorch.neuroevolution import SupervisedNE
import matplotlib.pyplot as plt
from evotorch.neuroevolution.net import count_parameters, parameter_vector
from time import time
from snntorch import spikegen
import math
from Common import *

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



num_epochs = 100



timestamp = str(int(time()))

LOGGER_FILE = 'logger_mixed_'+timestamp+'.txt'
NETWORK_NAME = 'mixed_'+timestamp+'.checkpoint'


net = SNN().to(device)
print(f'Network has {count_parameters(net)} parameters')

input()

train_dataset,test_dataset = load_fashion()






 


dataloading_time = 0
spikegen_time = 0
forward_time = 0
learning_time = 0
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

acc_hist = []
loss_hist = []
test_acc_vals = []
# training loop

start = time()

for epoch in range(num_epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)


        
    mnist_problem = SupervisedNE(
        train_dataset,  # Using the dataset specified earlier
        net,  # Training the SNN module designed earlier
        loss_fn,  # Minimizing CrossEntropyLoss
        #SF.ce_count_loss(),  # Minimizing CrossEntropyLoss
        minibatch_size = 32,  # With a minibatch size of 32
        num_minibatches = 1,
        common_minibatch = True,  # Always using the same minibatch across all solutions on an actor
        num_actors = 1,  # The total number of CPUs used
        num_gpus_per_actor = 'max',  # Dividing all available GPUs between the 4 actors
        device=device
        #subbatch_size = 50,  # Evaluating solutions in sub-batches of size 50 ensures we won't run out of GPU memory for individual workers
    )

    from evotorch.algorithms import PGPE,CMAES



    searcher = CMAES(mnist_problem,stdev_init=.1,separable=True , center_init =parameter_vector(net).to(device) )

    net = net.to(device)


    from evotorch.logging import StdOutLogger, PandasLogger
    stdout_logger = StdOutLogger(searcher, interval = 1)
    pandas_logger = PandasLogger(searcher, interval = 1)


    if epoch % 2 == 0:


        for i, (_data, targets) in enumerate(iter(trainloader)):
            

            #data = _data.transpose(0,1).to(device)
            data = _data.to(device)
            
            targets = targets.to(device)

            net.train()

            
            spk_rec = net(data)
            


            loss_val = loss_fn(spk_rec, targets)
            
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()


            
            # Store loss history for future plotting
            with torch.no_grad():
                
                
                loss_hist.append(loss_val.item())
                
                acc = SF.accuracy_rate(spk_rec, targets, population_code=True, num_classes=10)
                acc_hist.append(acc)

            if True:
                print(f"Epoch {epoch}, Iteration {i}/{len(trainloader)} \nTrain Loss: {loss_val.item():.2f} Accuracy: {acc * 100:.2f}%")

            if i%10 == 0:
                test_acc = evaluate(net,test_dataset)
                test_acc_vals += [test_acc]
                aux_logger = {}
                aux_logger["test_acc"] = test_acc_vals
                aux_logger["train_acc"] = acc_hist
                aux_logger["train_loss"] = loss_hist
                with open(LOGGER_FILE,"w") as f:
                    f.write(str(aux_logger))
                save(net,NETWORK_NAME)
    else:
        iterations = (len(train_dataset)//batch_size + 1)
        for k in range(0,iterations,10):
            searcher.run(10)

            final_sol = torch.squeeze(searcher.status["best"])
            
            print("SOL SIZE: ",final_sol.size())
            #input()
            net = mnist_problem.parameterize_net(final_sol).to(device)

            #loss = torch.nn.CrossEntropyLoss()
            acc = evaluate(net,test_dataset)
            test_acc_vals += [ acc]

            print(test_acc_vals)

            metrics = {}
            metrics['cmaes_logger'] = pandas_logger.to_dataframe()
            aux_logger = {}
            #aux_logger["iter"] = metrics['cmaes_logger']['iter'].tolist()
            aux_logger["stepsize"] = metrics['cmaes_logger']['stepsize'].tolist()
            aux_logger["mean_eval"] = metrics['cmaes_logger']["mean_eval"].tolist()
            aux_logger["median_eval"] = metrics['cmaes_logger']["median_eval"].tolist()
            aux_logger["pop_best_eval"] = metrics['cmaes_logger']["pop_best_eval"].tolist()
            aux_logger['test_acc'] = test_acc_vals
            #metrics['cmaes_logger'] = aux_logger
        
            with open(LOGGER_FILE,'w') as f:
                f.write(str(aux_logger))
            save(net,NETWORK_NAME)

print(f"Finished training after {time() - start} seconds.")

aux_logger["time"] = time() - start
final_test = evaluate(net,test_dataset)
aux_logger["final_test"] = final_test

with open(LOGGER_FILE,"w") as f:
    f.write(str(aux_logger))

save(net,NETWORK_NAME)
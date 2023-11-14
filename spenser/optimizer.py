import torch
from evotorch.algorithms import CMAES
from evotorch.neuroevolution import SupervisedNE
from evotorch.neuroevolution.net import count_parameters, parameter_vector
from math import log

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def assemble_optimizer(learning, model, dataset=None, config=None, loss_fn=None):
    problem = None
    print("IN OPTIMIZER")
    #print(Network.genotype_layers)
    #exit(0)
    if learning['learning'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr = float(learning['lr']),
                                    alpha = float(learning['rho']),
                                    weight_decay = float(learning['decay']))
        optimizer.is_backprop = True
        optimizer.is_cma_es = False
    elif learning['learning'] == 'gradient-descent':
        optimizer = torch.optim.SGD(model.parameters(),
                                lr = float(learning['lr']),
                                momentum = float(learning['momentum']),
                                weight_decay = float(learning['decay']),
                                nesterov = bool(learning['nesterov']))
        optimizer.is_backprop = True
        optimizer.is_cma_es = False
    elif learning['learning'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                lr = float(learning['lr']),
                                betas = tuple((float(learning['beta1']),float(learning['beta2']))),
                                weight_decay = float(learning['decay']),
                                amsgrad = bool(learning['amsgrad']))
        optimizer.is_backprop = True
        optimizer.is_cma_es = False
    elif learning['learning'] == 'cma-es':
        #N = count_parameters(model)
        #pop = int(4+3*log(N))
        
        #model.eval()
        problem = SupervisedNE( 
                        dataset,  # Using the dataset specified earlier
                        model,  # Training the SNN module designed earlier
                        loss_fn,  # Minimizing CrossEntropyLoss
                        minibatch_size = config["TRAINING"]["batch_size"],  # With a minibatch size of 32
                        #minibatch_size = 128,
                        num_minibatches = 1,
                        common_minibatch = True,  # Always using the same minibatch across all solutions on an actor
                        #num_actors = config["TRAINING"]["CMA-ES"]["num_actors"],  # The total number of CPUs used
                        num_actors = 1,
                        num_gpus_per_actor = 'max',  # Dividing all available GPUs between the 4 actors
                        #subbatch_size = config["TRAINING"]["CMA-ES"]["subbatch"],  # Evaluating solutions in sub-batches of size 50 ensures we won't run out of GPU memory for individual workers
                        device=device
                    )
        #print(learning['center_init'],type(learning['center_init']))

        #print(pop)
        optimizer = CMAES(problem,
                            stdev_init = float(learning['stdev_init']),
                            popsize = None,
                            #popsize = int(config["TRAINING"]["CMA-ES"]["popsize"]),
                            center_init=parameter_vector(model),
                            #center_learning_rate=float(learning['center_lr']),
                            #cov_learning_rate=float(learning['cov_lr']),
                            #rankmu_learning_rate=float(learning['rankmu_lr']),
                            #rankone_learning_rate=float(learning['rankone_lr']),
                            #stdev_min=learning['stdev_min'],
                            #stdev_max=learning['stdev_max'],
                            separable=learning['separable']
                        )
        optimizer.is_cma_es = True
        optimizer.is_backprop = False   
    return optimizer, problem
import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
from snntorch import spikegen
from time import time as t
from snntorch import functional as SF
from evotorch.neuroevolution.net.misc import fill_parameters
from spenser.optimizer import assemble_optimizer
DEBUG = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    
    
    def __init__(self,genotype_layers,input_size):
        super().__init__()
        #print(self.genotype_layers)
        #mem_share = MemShare()
        aux_layers = assemble_network(genotype_layers,input_size)
        self.tags = [l[0] for l in aux_layers]
        self.layers = nn.ModuleList([l[1] for l in aux_layers])
        #del aux_layers
        self.n_modules = len(self.layers)


    def forward(self, x):
        membrane_potentials = {}
        #Hacking with indexes because ModuleList does not support iteration

        for i in range(self.n_modules):
            tag = self.tags[i]
            layer = self.layers[i]
            if 'act' in tag:
                # Initialize hidden states at t=0
                membrane_potentials[tag] = layer.init_leaky() #TODO init_hidden ?

        
        # Record the final layer
        spk_rec = []
        
        #Not recording the membrane potential for now
        x = x.transpose(0,1)
        num_steps = x.size(0) #TODO
        #print(num_steps)
        #print(x.size())
        for T in range(num_steps):
            
            cur = x[T]
            
            for i in range(self.n_modules):
                tag = self.tags[i]
                layer = self.layers[i]
                if 'act' in tag:
                    cur,membrane_potentials[tag] = layer(cur,membrane_potentials[tag])
                else:
                    cur = layer(cur)
                if i == self.n_modules-1:
                    spk_rec.append(cur)

           
        #print(torch.stack(spk2_rec, dim=0))
        return torch.stack(spk_rec, dim=0)#, torch.stack(mem2_rec, dim=0)
    
  


def assemble_network(torch_layers,input_size):
    if DEBUG:
        #print("In assemble",torch_layers)
        pass
    last_output = input_size
    
    layers = []
    idx = 0
    
    first_fc = True

    grads_dict = {
        "atan": surrogate.atan(),
        "fast_sigmoid": surrogate.fast_sigmoid(),
        "triangular": surrogate.triangular()
    }

    for layer_type, layer_params in torch_layers:
        #Adding layers as tuple (string_id,layer) so that we can assemble them using Sequential(OrderededDict)


        if last_output[1] <= 0:
            return None
        if layer_type == 'act':
            spike_grad = grads_dict[layer_params["surr-grad"][0]]
            layer = snn.Leaky(beta=float(layer_params["beta"][0]),
                                threshold=float(layer_params["threshold"][0]),
                                init_hidden=False, #TODO
                                learn_beta=eval(layer_params["beta-trainable"][0]),
                                learn_threshold=eval(layer_params["threshold-trainable"][0]),
                                reset_mechanism=layer_params["reset"][0],
                                spike_grad=spike_grad)
            layers += [('act:'+str(idx),layer)]
            idx += 1

        elif layer_type == 'fc':
            if first_fc:
                layers += [(str(idx),nn.Flatten())]
                idx += 1
                first_fc = False
                #print("Flattening",last_output)
                last_output = (1,last_output[0] * last_output[1] * last_output[2])

            num_units = int(layer_params['num-units'][0])
            
            
            
            fc = nn.Linear(
                in_features=last_output[1], 
                out_features=num_units,
                bias=eval(layer_params['bias'][0]))
            
            layers += [(str(idx),fc)]
            idx += 1
            
            last_output = (1,num_units)

        elif layer_type == 'conv':
            
            W = last_output[1]
            NF = int(layer_params['num-filters'][0])
            K = int(layer_params['filter-shape'][0])
            S = int(layer_params['stride'][0])
            P = layer_params['padding'][0]
            if P == 'same':
                S = 1
            
            conv_layer = nn.Conv2d( in_channels=last_output[0],
                                    out_channels=NF,
                                    kernel_size=K,
                                    stride=S,
                                    padding=P,
                                    bias=eval(layer_params['bias'][0]))
            
            layers += [(str(idx),conv_layer)]
            idx += 1
        
            if P == 'valid':
                P = 0
                new_dim = int(((W - K + 2*P)/S) + 1)
            else:
                new_dim = last_output[1]
            last_output = (NF,new_dim,new_dim)


        elif layer_type == 'pool-max' or layer_type == 'pool-avg':
            K = int(layer_params['kernel-size'][0])

            if layer_type == 'pool-avg':
                pooling = nn.AvgPool2d(K)
            elif layer_type == 'pool-max':
                pooling = nn.MaxPool2d(K)

            layers += [(str(idx),pooling)]
            idx += 1

            new_dim = int(((last_output[1] - K) / K) + 1)
            last_output = (last_output[0], new_dim, new_dim)


        elif layer_type == 'dropout':
            rate = float(layer_params['rate'][0])
            dropout = nn.Dropout(p=rate)
            layers += [(str(idx),dropout)]
            idx += 1
        
        elif layer_type == 'no-op':
            #might be useful to collect metrics??
            pass

    layers[-1][1].output = True
    #model = nn.Sequential(OrderedDict(layers))
    #Network.tags = [l[0] for l in layers]
    #Network.modules = nn.ModuleList([l[1] for l in layers])
    #Network.n_modules = len(layers)
    #Network.layers = layers
    if DEBUG:
        print(layers)
    return layers


def forward_pass(net, data):
    '''
    spk_rec = []

    
    
    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
        
    '''    
    #utils.reset(net)  # resets hidden states for all LIF neurons in net
    spk_rec = net(data)
    return spk_rec



def get_fitness(model,testloader,num_steps):
    total = 0
    correct = 0
  
    
    with torch.no_grad():
        model.eval()
        for data, targets in testloader:
            #print("HEY!")
            #data = spikegen.rate(data.data, num_steps=num_steps).to(device)
            #data = data.transpose(0,1).to(device)
            data = data.to(device)
            targets = targets.to(device)
            
            
            spk_rec = forward_pass(model, data)
            
            #aux_spike_rec += list(spk_rec)
            # calculate total accuracy
            _, predicted = spk_rec.sum(dim=0).max(1)
            #acc = SF.accuracy_rate(spk_rec, targets)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    accuracy_test = correct / total

    if DEBUG:
        print(f"Total correctly classified test set images: {correct}/{total}")
        print(f"Test Set Accuracy: {100*accuracy_test:.2f}%")
    return accuracy_test

def train_with_cmaes(problem,optimizer,cmaes_iterations):
    dataloading_time = 0
    spikegen_time = 0
    forward_time = 0
    learning_time = 0
    start = t()
    if DEBUG:
        from evotorch.logging import StdOutLogger, PandasLogger
        stdout_logger = StdOutLogger(optimizer, interval = 1)
        pandas_logger = PandasLogger(optimizer, interval = 1)
    optimizer.run(cmaes_iterations)
    training_time = t()-start
    print(pandas_logger.to_dataframe())
    final_sol = torch.squeeze(optimizer.status["center"])
    print(final_sol.size())
    model = problem.parameterize_net(final_sol).to(device)
    #fill_parameters(model,final_sol)
    time_stats = {
        "training_time":training_time,
        "spikegen_time":spikegen_time,
        "forward_time":forward_time,
        "learning_time":learning_time,
        "dataloading_time":dataloading_time
    }
    if DEBUG:
        print("Training time (s): ",training_time)
    return model, time_stats, pandas_logger.to_dataframe()


def train_network(model,dataset,dataloader,optimizer_genotype,loss_fn,num_epochs=0,cmaes_iterations=0,config=None):
    optimizer,problem = assemble_optimizer(optimizer_genotype, model, dataset=dataset, config=config, loss_fn=loss_fn)
    acc_hist=[]
    loss_val=[]
    cmaes_logger=None
    
    
    if optimizer.is_backprop:
        acc_hist, loss_val, time_stats = train_with_backprop(model,dataloader,optimizer,loss_fn,num_epochs)
    

    elif optimizer.is_cma_es:
        del model
        model,time_stats,cmaes_logger = train_with_cmaes(problem,optimizer,cmaes_iterations)
        

    return model,acc_hist, loss_val, time_stats,cmaes_logger



def train_with_backprop(model,trainloader,optimizer,loss_fn,num_epochs):
    dataloading_time = 0
    spikegen_time = 0
    forward_time = 0
    learning_time = 0
    start = t()
    
    acc_hist = []
    loss_hist = []
    # training loop
    for epoch in range(num_epochs):
        for i, (_data, targets) in enumerate(iter(trainloader)):
            if DEBUG:
                
                if i%25 == 0:
                    print(f"\t[{i+1}/{num_epochs}] Current speed:{i/(t()-start)} iterations per second")
                
            a = t()
            #data = _data.transpose(0,1).to(device)
            data = _data.to(device)
            spikegen_time += t() - a

            '''
            (unique, counts) = np.unique(np.asarray(targets), return_counts=True)
            #print("EVO_Y_TEST:")
            print(np.asarray((unique, counts)).T)
            '''
            a = t()
         
            targets = targets.to(device)
 
            dataloading_time += t()-a
            
            model.train()
            a = t()
            
            spk_rec = forward_pass(model, data)
            
            forward_time += t() - a
            
            a = t()
            loss_val = loss_fn(spk_rec, targets)
         
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            learning_time += t() - a

            
            # Store loss history for future plotting
            with torch.no_grad():
                loss_hist.append(loss_val.item())
               
                acc = SF.accuracy_rate(spk_rec, targets)
                acc_hist.append(acc)

            if DEBUG:
                print(f"Epoch {epoch}, Iteration {i}/{len(trainloader)} \nTrain Loss: {loss_val.item():.2f} Accuracy: {acc * 100:.2f}%")

    training_time = t()-start
    dataloading_time = training_time - forward_time - learning_time - spikegen_time
    
    time_stats = {
        "training_time":training_time,
        "spikegen_time":spikegen_time,
        "forward_time":forward_time,
        "learning_time":learning_time,
        "dataloading_time":dataloading_time
    }
    if DEBUG:
        print("Training time (s): ",training_time)
        print("Time spent converting dataset (s / %): ",spikegen_time,100*spikegen_time/training_time)
        print("Time spent in forward pass (s / %):",forward_time,100*forward_time/training_time)
        print("Time spent in learning (s / %)",learning_time,100*learning_time/training_time)
        print("Time spent loading data (s / %):", dataloading_time,100*dataloading_time/training_time)
    return acc_hist, loss_hist, time_stats

from spenser.data import load_dataset
from time import time as t

import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from typing import OrderedDict
from snntorch import spikegen

DEBUG = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Evaluator:
    """
        Stores the dataset, maps the phenotype into a trainable model, and
        evaluates it


        Attributes
        ----------
        dataset : dict
            dataset instances and partitions

        fitness_metric : function
            fitness_metric (y_true, y_pred)
            y_pred are the confidences


        Methods
        -------
        get_layers(phenotype)
            parses the phenotype corresponding to the layers
            auxiliary function of the assemble_network function

        get_learning(learning)
            parses the phenotype corresponding to the learning
            auxiliary function of the assemble_optimiser function

        assemble_network(keras_layers, input_size)
            maps the layers phenotype into a keras model

        assemble_optimiser(learning)
            maps the learning into a keras optimiser

        evaluate(phenotype, load_prev_weights, weights_save_path, parent_weights_path,
                 train_time, num_epochs, datagen=None, input_size=(32, 32, 3))
            evaluates the keras model using the keras optimiser

        testing_performance(self, model_path)
            compute testing performance of the model
    """

    def __init__(self, dataset, config):
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        self.config = config
        self.dataset = load_dataset(dataset,config)
        


    def get_layers(self, phenotype):
        """
            Parses the phenotype corresponding to the layers.
            Auxiliary function of the assemble_network function.

            Parameters
            ----------
            phenotye : str
                individual layers phenotype

            Returns
            -------
            layers : list
                list of tuples (layer_type : str, node properties : dict)
        """

        raw_phenotype = phenotype.split(' ')

        idx = 0
        first = True
        node_type, node_val = raw_phenotype[idx].split(':')
        layers = []

        while idx < len(raw_phenotype):
            if node_type == 'layer':
                if not first:
                    layers.append((layer_type, node_properties))
                else:
                    first = False
                layer_type = node_val
                node_properties = {}
            else:
                node_properties[node_type] = node_val.split(',')

            idx += 1
            if idx < len(raw_phenotype):
                node_type, node_val = raw_phenotype[idx].split(':')

        layers.append((layer_type, node_properties))

        return layers

    def get_learning(self, learning):
        """
            Parses the phenotype corresponding to the learning
            Auxiliary function of the assemble_optimiser function

            Parameters
            ----------
            learning : str
                learning phenotype of the individual

            Returns
            -------
            learning_params : dict
                learning parameters
        """

        raw_learning = learning.split(' ')

        idx = 0
        learning_params = {}
        while idx < len(raw_learning):
            param_name, param_value = raw_learning[idx].split(':')
            learning_params[param_name] = param_value.split(',')
            idx += 1

        for _key_ in sorted(list(learning_params.keys())):
            if len(learning_params[_key_]) == 1:
                try:
                    learning_params[_key_] = eval(learning_params[_key_][0])
                except NameError:
                    learning_params[_key_] = learning_params[_key_][0]

        return learning_params


    def assemble_network(self, torch_layers, input_size):
        if DEBUG:
            print(torch_layers)
        
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
                                  init_hidden=True,
                                  learn_beta=eval(layer_params["beta-trainable"][0]),
                                  learn_threshold=eval(layer_params["threshold-trainable"][0]),
                                  reset_mechanism=layer_params["reset"][0],
                                  spike_grad=spike_grad)
                layers += [(str(idx),layer)]
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
        model = nn.Sequential(OrderedDict(layers))
        del layers
        if DEBUG:
            print(model)
        return model

    def assemble_optimiser(self, learning, model):
        """
            Maps the learning into a keras optimiser

            Parameters
            ----------
            learning : dict
                output of get_learning

            Returns
            -------
            optimiser : torch.optim.Optimizer
                torch optimiser that will be later used to train the model
        """

        if learning['learning'] == 'rmsprop':
            return torch.optim.RMSprop(model.parameters(),
                                        lr = float(learning['lr']),
                                        alpha = float(learning['rho']),
                                        weight_decay = float(learning['decay']))
        
        elif learning['learning'] == 'gradient-descent':
            return torch.optim.SGD(model.parameters(),
                                   lr = float(learning['lr']),
                                   momentum = float(learning['momentum']),
                                   weight_decay = float(learning['decay']),
                                   nesterov = bool(learning['nesterov']))

        elif learning['learning'] == 'adam':
            return torch.optim.Adam(model.parameters(),
                                    lr = float(learning['lr']),
                                    betas = tuple((float(learning['beta1']),float(learning['beta2']))),
                                    weight_decay = float(learning['decay']),
                                    amsgrad = bool(learning['amsgrad']))


    def save(self, model, optimizer, loss_fn, torch_layers, torch_learning, input_size, weights_save_path):
        # Save the models phenotype and weights 
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_layers": torch_layers,
            "torch_learning": torch_learning,
            "input_size": input_size,
            "loss_fn": loss_fn
        }
        torch.save(checkpoint, weights_save_path + '.checkpoint')
        
    
    def load(self,weights_save_path):

        checkpoint = torch.load(weights_save_path + '.checkpoint')

        input_size = checkpoint["input_size"]
        torch_layers = checkpoint["torch_layers"]
        torch_learning = checkpoint["torch_learning"]
        loss_fn = checkpoint["loss_fn"]


        model = self.assemble_network(torch_layers, input_size)
        optimizer = self.assemble_optimiser(torch_learning,model)

        
        model.to(device)
        #print(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, loss_fn, torch_layers, torch_learning

    def testing_performance(self, weights_save_path):
        testloader = self.dataset["test"]
        num_steps = int(self.config["TRAINING"]["num_steps"])
        model, optimizer, loss_fn, torch_layers, torch_learning = self.load(weights_save_path)
        #model.to(device)
        accuracy_test = get_fitness(model,testloader,num_steps)
        return accuracy_test

    def retrain_longer(self, weights_save_path, num_epochs, save_path):

        
        model, optimizer, loss_fn, torch_layers, torch_learning = self.load(weights_save_path)
        #model.to(device)

        trainloader = self.dataset["evo_train"]
        testloader = self.dataset["evo_test"]

        
        input_size = self.dataset["input_size"]
        #input_size = (1,28,28)
        loss_hist = []
        acc_hist = []
        history = {}
        history['accuracy'] = []
        history['loss'] = []
        history['time_stats'] = []
        num_steps = int(self.config["TRAINING"]["num_steps"])
        
        

        for epoch in range(num_epochs):
            
            acc_hist, loss_hist, time_stats = train_network(model,trainloader,optimizer,loss_fn,1,num_steps)
            history['accuracy'] += acc_hist
            history['loss'] += loss_hist
            history['time_stats'] += [time_stats]


            acc_hist, loss_hist, time_stats = train_network(model,testloader,optimizer,loss_fn,1,num_steps)
            history['accuracy'] += acc_hist
            history['loss'] += loss_hist
            history['time_stats'] += [time_stats]

        self.save(model, optimizer, loss_fn, torch_layers, torch_learning, input_size, save_path)
        return history

    def evaluate(self, phenotype, weights_save_path, parent_weights_path,\
                num_epochs): #pragma: no cover
        
        start = t()
        trainloader = self.dataset["evo_train"]
        testloader = self.dataset["evo_test"]

        

        loss_hist = []
        acc_hist = []
        history = {}

        num_steps = int(self.config["TRAINING"]["num_steps"])
        input_size = self.dataset["input_size"]
        #input_size = (1,28,28)
        spk_rec = []

        time_stats = {}

        model = None
        optimizer = None
        loss_fn = None
        loss_val = None   
        
        if num_epochs > 0:
            model_phenotype, learning_phenotype = phenotype.split('learning:')
            
            learning_phenotype = 'learning:'+learning_phenotype.rstrip().lstrip()
            model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

            torch_layers = self.get_layers(model_phenotype)
            torch_learning = self.get_learning(learning_phenotype)
        
            
            model = self.assemble_network(torch_layers, input_size)
            if model == None:
                return None

            model.to(device)

            optimizer = self.assemble_optimiser(torch_learning,model)
                            
            loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0)
            #loss_fn = SF.ce_rate_loss()
            #print("Rate loss")
            acc_hist, loss_hist, time_stats = train_network(model,trainloader,optimizer,loss_fn,num_epochs,num_steps)
            
            history['accuracy'] = acc_hist
            history['loss'] = loss_hist
            
        
        else:
            #Is parent, already trained
            model, optimizer, loss_fn, torch_layers, torch_learning = self.load(parent_weights_path)
            model.to(device)

        self.save(model, optimizer, loss_fn, torch_layers, torch_learning, input_size, weights_save_path)
        
        #measure test performance
        start_fitness = t()


        accuracy_test = get_fitness(model,testloader,num_steps)
        #accuracy_test = 0.8
        if DEBUG:
            print("Exited get_fitness")

        time_stats["fitness_time"] = t() - start_fitness
        time_stats["total_time"] = t()-start

        history['accuracy_test'] = accuracy_test
        history['time_stats'] = time_stats

        total_params = sum(p.numel() for p in model.parameters())

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        history['trainable_parameters'] = total_trainable_params
        history['total_parameters'] = total_params
        if DEBUG:
            print("No. of parameters collected.")


        return history




def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
        
        

    return torch.stack(spk_rec)

def get_fitness(model,testloader,num_steps):
    total = 0
    correct = 0
  
    
    with torch.no_grad():
        model.eval()
        for data, targets in testloader:
            
            data = spikegen.rate(data.data, num_steps=num_steps).to(device)
            #data = data.to(device)
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

def train_network(model,trainloader,optimizer,loss_fn,num_epochs,num_steps):
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
          
            data = spikegen.rate(_data.data, num_steps=num_steps).to(device)

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

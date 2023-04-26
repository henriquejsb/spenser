
# Copyright 2019 Filipe Assuncao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
import random
from time import time
import numpy as np
import os
from fast_denser.utilities.data import load_dataset
import gc
import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from typing import OrderedDict
from torch.utils.data import DataLoader
#import tonic
from snntorch import spikegen

from multiprocessing import Pool
from multiprocessing import Queue
from multiprocessing import set_start_method
from multiprocessing import Process
import contextlib
from time import time as t
#from tqdm import tqdm


import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

DEBUG = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_num_threads(os.cpu_count() - 1)

if DEBUG:
    #print("Running on Device = ", device)
    pass

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
        #last_output = input_size[0]*input_size[1]
        last_output = input_size
        
        layers = []
        idx = 0
        #beta = 0.9  # neuron decay rate
        #spike_grad = surrogate.fast_sigmoid()
        first_fc = True

        grads_dict = {
            "atan": surrogate.atan(),
            "fast_sigmoid": surrogate.fast_sigmoid(),
            "triangular": surrogate.triangular()
        }

        for layer_type, layer_params in torch_layers:
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
                
                #print(layer_params)
                #input()
                #Adding layers as tuple (string_id,layer) so that we can assemble them using Sequential(OrderededDict)
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

            elif layer_type == 'pool-max':
                pass


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
        #input()
        '''
        #-------------------------------------------------------------

        #In case we need to load the module we need phenotype and weights
        
        model = None
        gc.collect()

        model,optimizer,loss_fn = self.load(weights_save_path)
        
        #model = torch.jit.load(weights_save_path)
        model.to(device)

        get_fitness(model,testloader,num_steps)
        #---------------------------------------------------
        '''

            
        #gc.collect()
        #torch.cuda.empty_cache()
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
 
            dataloading_time += time()-a
            
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
            learning_time += time() - a

            
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

def evaluate(args): #pragma: no cover
    """
        Function used to deploy a new process to train a candidate solution.
        Each candidate solution is trained in a separe process to avoid memory problems.

        Parameters
        ----------
        args : tuple
            cnn_eval : Evaluator
                network evaluator

            phenotype : str
                individual phenotype

            load_prev_weights : bool
                resume training from a previous train or not

            weights_save_path : str
                path where to save the model weights after training

            parent_weights_path : str
                path to the weights of the previous training

            train_time : float
                maximum training time

            num_epochs : int
                maximum number of epochs

        Returns
        -------
        score_history : dict
            training data: loss and accuracy
    """

    

    return_val, scnn_eval, phenotype, weights_save_path, parent_weights_path, num_epochs = args
    
    try:
        history = scnn_eval.evaluate(phenotype, weights_save_path, parent_weights_path, num_epochs)
        
    except KeyboardInterrupt:
        # quit
        exit(0)
    except torch.cuda.OutOfMemoryError as e:
        traceback.print_exc()
        print("CUDA ERROR")
        history = None
    except RuntimeError as e:
        if 'Unable to find a valid cuDNN algorithm' in traceback.format_exc():
            print("CUDA ERROR")
            traceback.print_exc()
            history = None
        else:
            exit(-5)
    if DEBUG:
        print("Returning from evaluate.")
    return_val.put(history)

    


class Module:
    """
        Each of the units of the outer-level genotype


        Attributes
        ----------
        module : str
            non-terminal symbol

        min_expansions : int
            minimum expansions of the block

        max_expansions : int
            maximum expansions of the block

        levels_back : dict
            number of previous layers a given layer can receive as input

        layers : list
            list of layers of the module

        connections : dict
            list of connetions of each layer


        Methods
        -------
            initialise(grammar, reuse)
                Randomly creates a module
    """

    def __init__(self, module, min_expansions, max_expansions):
        """
            Parameters
            ----------
            module : str
                non-terminal symbol

            min_expansions : int
                minimum expansions of the block
        
            max_expansions : int
                maximum expansions of the block

        """

        self.module = module
        self.min_expansions = min_expansions
        self.max_expansions = max_expansions
        self.layers = []

    def initialise(self, grammar, reuse, init_max):
        """
            Randomly creates a module

            Parameters
            ----------
            grammar : Grammar
                grammar instace that stores the expansion rules

            reuse : float
                likelihood of reusing an existing layer

            Returns
            -------
            score_history : dict
                training data: loss and accuracy
        """

        num_expansions = random.choice(init_max[self.module])

        #Initialise layers
        for idx in range(num_expansions):
            if idx>0 and random.random() <= reuse:
                r_idx = random.randint(0, idx-1)
                self.layers.append(self.layers[r_idx])
            else:
                self.layers.append(grammar.initialise(self.module))

       



class Individual:
   
    def __init__(self, network_structure, macro_rules, output_rule, ind_id):
     

        self.network_structure = network_structure
        self.output_rule = output_rule
        self.macro_rules = macro_rules
        self.modules = []
        self.output = None
        self.macro = []
        self.phenotype = None
        self.fitness = None
        self.metrics = None
        self.num_epochs = 0
        self.trainable_parameters = None
        self.total_parameters = None
        self.time = None
        self.current_time = 0
        self.train_time = 0
        self.id = ind_id
        self.is_parent = False

    def initialise(self, grammar, reuse, init_max):
        """
            Randomly creates a candidate solution

            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules


            reuse : float
                likelihood of reusing an existing layer

            Returns
            -------
            candidate_solution : Individual
                randomly created candidate solution
        """

        for non_terminal, min_expansions, max_expansions in self.network_structure:
            new_module = Module(non_terminal, min_expansions, max_expansions)
            new_module.initialise(grammar, reuse, init_max)

            self.modules.append(new_module)

        #Initialise output
        self.output = grammar.initialise(self.output_rule)

        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.initialise(rule))

        return self


    def decode(self, grammar):
        """
            Maps the genotype to the phenotype

            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules

            Returns
            -------
            phenotype : str
                phenotype of the individual to be used in the mapping to the keras model.
        """

        phenotype = ''
        offset = 0
        layer_counter = 0
        for module in self.modules:
            offset = layer_counter
            for layer_idx, layer_genotype in enumerate(module.layers):
                layer_counter += 1
                phenotype += ' ' + grammar.decode(module.module, layer_genotype)

        phenotype += ' '+grammar.decode(self.output_rule, self.output)
     
        for rule_idx, macro_rule in enumerate(self.macro_rules):
            phenotype += ' '+grammar.decode(macro_rule, self.macro[rule_idx])

        self.phenotype = phenotype.rstrip().lstrip()
        return self.phenotype

    def evaluate(self, grammar, cnn_eval, weights_save_path, parent_weights_path=''): #pragma: no cover
  
        if DEBUG:
            print(torch.cuda.memory_allocated(),torch.cuda.max_memory_allocated())
        
        phenotype = self.decode(grammar)
        return_val = Queue()
        
        #print(f"Begin training individual {self.id}.\n")
        process = Process(target=evaluate, args=[(return_val, cnn_eval, phenotype,
                            weights_save_path, parent_weights_path,\
                            self.num_epochs)])
        # run the process
        process.start()
        # wait for the process to finish
        #print('Waiting for the process...')
        process.join()
        if DEBUG:
            print("Process terminated")
        metrics = return_val.get()
        
        '''
        num_pool_workers=1 
        set_start_method('spawn')
        
        with contextlib.closing(Pool(num_pool_workers)) as po: 
        #metrics = evaluate((cnn_eval, phenotype,
        #                    weights_save_path, parent_weights_path,\
        #                    self.num_epochs))
            pool_results = po.map_async(evaluate,[(cnn_eval, phenotype,
                            weights_save_path, parent_weights_path,\
                            self.num_epochs)])
            metrics = pool_results.get()[0]
        '''
        
        if not self.metrics:
            self.metrics = {}
        
        if metrics is not None:
            if self.num_epochs == 0 and self.metrics is not None:
                if 'accuracy_test' in metrics:
                    if type(metrics['accuracy_test']) is float:
                        self.fitness = metrics['accuracy_test']
                    else:
                        self.fitness = metrics['accuracy_test'].item()
            else:
                if 'accuracy' in metrics:
                    if type(metrics['accuracy']) is list:
                        self.metrics['accuracy'] = [i for i in metrics['accuracy']]
                    else:
                        self.metrics['accuracy'] = [i.item() for i in metrics['accuracy']]
                
                if 'loss' in metrics:
                    if type(metrics['loss']) is list:
                        self.metrics['loss'] = [i for i in metrics['loss']]
                    else:
                        self.metrics['loss'] = [i.item() for i in metrics['loss']]


                #self.metrics = metrics

                if 'accuracy_test' in metrics:
                    if type(metrics['accuracy_test']) is float:
                        self.fitness = metrics['accuracy_test']
                    else:
                        self.fitness = metrics['accuracy_test'].item()
                if 'time_stats' in metrics:
                    time_stats = metrics['time_stats']
                    self.metrics['time_stats'] = time_stats
                    if 'total_time' in time_stats:
                        self.time = time_stats['total_time']
                    if 'training_time' in time_stats:
                        self.train_time = time_stats['training_time']
                self.trainable_parameters = metrics['trainable_parameters']
                self.total_parameters = metrics['total_parameters']

            #del metrics
        else:
            self.metrics = None
            self.fitness = -1
            self.num_epochs = 0
            self.trainable_parameters = -1
            self.current_time = 0
        
        # print("Finished. Press to see garbage")
        # input()
        # for obj in gc.get_objects():
        #         try:
        #             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #                 print(obj,type(obj), obj.size())
        #         except:
        #             pass
        # #print("Garbage finished")
        #input()
        return self.fitness


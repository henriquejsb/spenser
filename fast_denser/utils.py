
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
import tonic
from snntorch import spikegen


from multiprocessing import Pool
import contextlib
from time import time as t
from tqdm import tqdm

DEBUG = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count() - 1)

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




    def assemble_network(self, torch_layers, input_size):

        #last_output = input_size[0]*input_size[1]
        last_output = (1,28,28)
        #TODO 
        layers = []
        idx = 0
        beta = 0.9  # neuron decay rate
        spike_grad = surrogate.fast_sigmoid()
        first_fc = True

        for layer_type, layer_params in torch_layers:
            if layer_type == 'fc':
                if first_fc:
                    layers += [(str(idx),nn.Flatten())]
                    idx += 1
                    first_fc = False
                    last_output = last_output[0] * last_output[1] * last_output[2]
                num_units = int(layer_params['num-units'][0])
                
                #Adding layers as tuple (string_id,layer) so that we can assemble them using Sequential(OrderededDict)
                fc = nn.Linear(last_output, num_units)
                activation = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
                
                layers += [(str(idx),fc)]
                idx += 1
                layers += [(str(idx),activation)]
                idx += 1
                last_output = num_units

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
                
                activation = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
                layers += [(str(idx),conv_layer)]
                idx += 1
                layers += [(str(idx),activation)]
                idx += 1
                if P == 'valid':
                    P = 0
                    new_dim = int(((W - K + 2*P)/S) + 1)
                else:
                    new_dim = last_output[1]
                last_output = (NF,new_dim,new_dim)

            elif layer_type == 'batch-norm':
                pass
            elif layer_type == 'pool-avg':
                K = int(layer_params['kernel-size'][0])
                pooling = nn.AvgPool2d(K)
                layers += [(str(idx),conv_layer)]
                idx += 1

                new_dim = last_output[1] - K + 1
                last_output = (last_output[0], new_dim, new_dim)
            elif layer_type == 'pool-max':
                pass
            elif layer_type == 'dropout':
                rate = float(layer_params['rate'][0])
                dropout = nn.Dropout(p=rate)
                layers += [(str(idx),dropout)]
                idx += 1
        layers[-1][1].output = True
        model = nn.Sequential(OrderedDict(layers))
        print(model)
        return model



    def evaluate(self, phenotype, weights_save_path, parent_weights_path,\
                num_epochs, input_size=(90, 90, 1)): #pragma: no cover
        
        model_phenotype = phenotype
        model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

        torch_layers = self.get_layers(model_phenotype)
        
        
        print("Running on Device = ", device)
        
        model = self.assemble_network(torch_layers, input_size)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-2, betas=(0.9, 0.999))
        loss_fn = SF.mse_count_loss(correct_rate=0.9, incorrect_rate=0.1)
        #batch_size = 128

        #trainloader = DataLoader(self.dataset, batch_size = batch_size, collate_fn = tonic.collation.PadTensors(), shuffle = True)
        trainloader = self.dataset["evo_train"]
        testloader = self.dataset["evo_test"]
        print(f"Begin training individual.\n")
        start = t()


        loss_hist = []
        acc_hist = []
        history = {}

        num_steps = 100


        def forward_pass(net, data):
            spk_rec = []
            utils.reset(net)  # resets hidden states for all LIF neurons in net
            #data = data.transpose(0,1)
            #print("IN FUNCTION ", data.shape)
            for step in range(data.size(0)):  # data.size(0) = number of time steps
                #print(data[step].shape)
                spk_out, mem_out = net(data[step])
                spk_rec.append(spk_out)
                #print(spk_out.size())
                #print(mem_out.size())
            return torch.stack(spk_rec)

        # training loop
        for epoch in range(num_epochs):
            for i, (data, targets) in enumerate(iter(trainloader)):

                data = spikegen.rate(data.data, num_steps=num_steps).to(device)
                targets = targets.to(device)
                #print(data.shape)

                #print(targets.shape)
                model.train()
                #spk_rec = forward_pass(net, data)
                spk_rec = forward_pass(model, data)
                #print("IN LOOP ",data.shape)
                #print("TARGETS",targets.shape)
                loss_val = loss_fn(spk_rec, targets)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

                #print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

                acc = SF.accuracy_rate(spk_rec, targets)
                acc_hist.append(acc)
                #print(f"Accuracy: {acc * 100:.2f}%\n")

        history['accuracy'] = acc_hist
        history['loss'] = loss_hist


        #save final model to file
        #model.save(weights_save_path.replace('.hdf5', '.h5'))
        torch.save(model.state_dict(),weights_save_path.replace('.hdf5', '.h5'))

        

        #measure test performance
        total = 0
        correct = 0
        aux_spike_rec = []
        aux_targets = []
        
        with torch.no_grad():
            model.eval()
            for data, targets in testloader:
                data = spikegen.rate(data.data, num_steps=num_steps).to(device)
                targets = targets.to(device)
                aux_targets += list(targets)
                # forward pass
                spk_rec = forward_pass(model, data)
                aux_spike_rec += list(spk_rec)
                # calculate total accuracy
                _, predicted = spk_rec.sum(dim=0).max(1)
                #acc = SF.accuracy_rate(spk_rec, targets)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy_test = correct / total
        print(f"Total correctly classified test set images: {correct}/{total}")
        print(f"Test Set Accuracy: {100*accuracy_test:.2f}%")
        history['accuracy_test'] = accuracy_test
        
        '''
        In case we need to load the module we need phenotype and weights
        
        model = None
        gc.collect()

        model = self.assemble_network(torch_layers, input_size)
        model.to(device)
        model.load_state_dict(torch.load(weights_save_path.replace('.hdf5', '.h5')))
        '''


        #Cleaning up
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        return history

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

    

    scnn_eval, phenotype, weights_save_path, parent_weights_path, num_epochs = args

    return scnn_eval.evaluate(phenotype, weights_save_path, parent_weights_path, num_epochs)
    


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
   
    def __init__(self, network_structure, output_rule, ind_id):
     

        self.network_structure = network_structure
        self.output_rule = output_rule
        #self.macro_rules = macro_rules
        self.modules = []
        self.output = None
        self.macro = []
        self.phenotype = None
        self.fitness = None
        self.metrics = None
        self.num_epochs = 0
        self.trainable_parameters = None
        self.time = None
        self.current_time = 0
        self.train_time = 0
        self.id = ind_id

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
     


        self.phenotype = phenotype.rstrip().lstrip()
        return self.phenotype

    def evaluate(self, grammar, cnn_eval, weights_save_path, parent_weights_path=''): #pragma: no cover
  

        phenotype = self.decode(grammar)
        start = time()

        print(f"Begin training individual {self.id}.\n")
        '''
        num_pool_workers=1 
        with contextlib.closing(Pool(num_pool_workers)) as po: 
            pool_results = po.map_async(evaluate, [(cnn_eval, phenotype,
                            weights_save_path, parent_weights_path,\
                            self.num_epochs)])
            metrics = pool_results.get()[0]
        '''
        metrics = evaluate((cnn_eval, phenotype,
                            weights_save_path, parent_weights_path,\
                            self.num_epochs))
        
        

        if metrics is not None:
            if 'accuracy' in metrics:
                if type(metrics['accuracy']) is list:
                    metrics['accuracy'] = [i for i in metrics['accuracy']]
                else:
                    metrics['accuracy'] = [i.item() for i in metrics['accuracy']]
            
            self.metrics = metrics

            if 'accuracy_test' in metrics:
                if type(self.metrics['accuracy_test']) is float:
                    self.fitness = self.metrics['accuracy_test']
                else:
                    self.fitness = self.metrics['accuracy_test'].item()
        
        return self.fitness



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

import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from typing import OrderedDict
from torch.utils.data import DataLoader
import tonic

from multiprocessing import Pool
import contextlib
from time import time as t
from tqdm import tqdm

DEBUG = False


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
        last_output = 34*34*2
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
                num_units = int(layer_params['num-units'][0])
                #Adding layers as tuple (string_id,layer) so that we can assemble them using Sequential(OrderededDict)
                layers += [(str(idx),nn.Linear(last_output, num_units))]
                idx += 1
                layers += [(str(idx),snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True))]
                idx += 1
                last_output = num_units
            elif layer_type == 'conv':
                pass
            elif layer_type == 'batch-norm':
                pass
            elif layer_type == 'pool-avg':
                pass
            elif layer_type == 'pool-max':
                pass
            elif layer_type == 'dropout':
                pass
        layers[-1][1].output = True
        model = nn.Sequential(OrderedDict(layers))
        print(model)
        return model

    def evaluate(self, phenotype, weights_save_path, parent_weights_path,\
                num_epochs, input_size=(90, 90, 1)): #pragma: no cover
        
        model_phenotype = phenotype
        model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

        torch_layers = self.get_layers(model_phenotype)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(os.cpu_count() - 1)
        print("Running on Device = ", device)
        model = self.assemble_network(torch_layers, input_size)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-2, betas=(0.9, 0.999))
        loss_fn = SF.mse_count_loss(correct_rate=0.9, incorrect_rate=0.1)
        batch_size = 128

        trainloader = DataLoader(self.dataset, batch_size = batch_size, collate_fn = tonic.collation.PadTensors(), shuffle = True)

        print("Begin training.\n")
        start = t()


        loss_hist = []
        acc_hist = []

        def forward_pass(net, data):
            spk_rec = []
            utils.reset(net)  # resets hidden states for all LIF neurons in net
            data = data.transpose(0,1)
            #print("IN FUNCTION ", data.shape)
            for step in range(data.size(0)):  # data.size(0) = number of time steps
                #print(data[step].shape)
                spk_out, mem_out = net(data[step])
                spk_rec.append(spk_out)

            return torch.stack(spk_rec)

        # training loop
        for epoch in range(num_epochs):
            for i, (data, targets) in enumerate(iter(trainloader)):
                data = data.to(device)
                targets = targets.to(device)
                print(data.shape)
                #print(targets.shape)
                model.train()
                #spk_rec = forward_pass(net, data)
                spk_rec = forward_pass(model, data)
                #print("IN LOOP ",data.shape)
                loss_val = loss_fn(spk_rec, targets)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

                print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

                acc = SF.accuracy_rate(spk_rec, targets)
                acc_hist.append(acc)
                print(f"Accuracy: {acc * 100:.2f}%\n")



        return {'accuracy_test':10}

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


        '''
        num_pool_workers=1 
        with contextlib.closing(Pool(num_pool_workers)) as po: 
            pool_results = po.map_async(evaluate, [(cnn_eval, phenotype,
                            weights_save_path, parent_weights_path,\
                            self.num_epochs)])
            metrics = pool_results.get()[0]
        '''
        self.fitness = evaluate((cnn_eval, phenotype,
                            weights_save_path, parent_weights_path,\
                            self.num_epochs))
       
        return self.fitness


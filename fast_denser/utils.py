
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
from bindsnet.network import Network
from bindsnet.learning import PostPre
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, Conv1dConnection, SparseConnection
from multiprocessing import Pool
import contextlib
from time import time as t

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




    def assemble_network(self, bindsnet_layers, input_size):
  
        #create Network object
        network = Network()
        #input layer
        inputs = Input(n=28 * 28, shape=(1, 28 * 28), traces=True)
        

        #Create layers -- ADD NEW LAYERS HERE
        layers = [(inputs,None)]
        for layer_type, layer_params in bindsnet_layers:
            #convolutional layer
            if layer_type == 'conv':
                filters = int(layer_params['num-filters'][0])
                kernel_size = int(layer_params['kernel-size'][0])
                padding = layer_params['padding'][0]
                strides = int(layer_params['strides'][0])
                conv_size = int((28 * 28 - kernel_size + 2 * padding) / strides) + 1

                conv_layer = LIFNodes(
                     n=filters * conv_size, 
                     shape=(filters, conv_size), 
                     traces=True
                )
                
                conv_conn = Conv1dConnection(
                    layers[-1][0],
                    conv_layer,
                    kernel_size=kernel_size,
                    stride=strides,
                    update_rule=PostPre,
                    norm=0.4 * kernel_size,
                    nu=[1e-4, 1e-2],
                    wmax=1.0,
                )
                layers.append((conv_layer, conv_conn))
            elif layer_type == 'fc':
                fc_layer = LIFNodes(
                    n=int(layer_params['num-units'][0]),
                    traces=True
                )
                fc_conn = Connection(
                    source=layers[-1][0],
                    target=fc_layer,
                    update_rule=PostPre,
                )
                layers.append((fc_layer, fc_conn))
            elif layer_type == 'sparse':
                fc_layer = LIFNodes(
                    n=int(layer_params['num-units'][0]),
                    traces=True
                )
                fc_conn = SparseConnection(
                    source=layers[-1][0],
                    target=fc_layer
                )
                layers.append((fc_layer, fc_conn))
         
            #END ADD NEW LAYERS

        network.add_layer(inputs, name="X")
        last_layer = "X"
        for ind,layer_and_conn in enumerate(layers[1:-1]):
            layer,conn = layer_and_conn
            network.add_layer(layer, name=str(ind))               
            network.add_connection(conn,source=last_layer, target=str(ind))
            last_layer = str(ind)


        network.add_layer(layers[-1][0], name="Y")
        network.add_connection(layers[-1][1],source=last_layer,target="Y")
        
        return network

    def evaluate(self, phenotype, weights_save_path, parent_weights_path,\
                num_epochs, input_size=(90, 90, 1)): #pragma: no cover
        
        model_phenotype = phenotype
        model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

        bindsnet_layers = self.get_layers(model_phenotype)
       

        model = self.assemble_network(bindsnet_layers, input_size)

        #save final moodel to file
        #model.save(weights_save_path.replace('.hdf5', '.h5'))
        print("Begin training.\n")
        start = t()
        #measure test performance
        
        
        if DEBUG:
            print(phenotype, accuracy_test)

        #score.history['trainable_parameters'] = trainable_count
        #score.history['accuracy_test'] = accuracy_test


        return {'accuracy':10}

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

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
   
    def __init__(self, network_structure, ind_id):
     

        self.network_structure = network_structure
        #self.output_rule = output_rule
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


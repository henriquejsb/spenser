
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
from multiprocessing import Pool
import contextlib

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
        self.dataset = load_dataset(dataset,config["DATASET"])
      


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




    def assemble_network(self, keras_layers, input_size):
        """
            Maps the layers phenotype into a keras model

            Parameters
            ----------
            keras_layers : list
                output from get_layers

            input_size : tuple
                network input shape

            Returns
            -------
            model : keras.models.Model
                keras trainable model
        """

        #input layer
        inputs = keras.layers.Input(shape=input_size)

        #Create layers -- ADD NEW LAYERS HERE
        layers = []
        for layer_type, layer_params in keras_layers:
            #convolutional layer
            if layer_type == 'conv':
                conv_layer = keras.layers.Conv2D(filters=int(layer_params['num-filters'][0]),
                                                 kernel_size=(int(layer_params['filter-shape'][0]), int(layer_params['filter-shape'][0])),
                                                 strides=(int(layer_params['stride'][0]), int(layer_params['stride'][0])),
                                                 padding=layer_params['padding'][0],
                                                 activation=layer_params['act'][0],
                                                 use_bias=eval(layer_params['bias'][0]),
                                                 kernel_initializer='he_normal',
                                                 kernel_regularizer=keras.regularizers.l2(0.0005))
                layers.append(conv_layer)

         
            #END ADD NEW LAYERS


       
        
        #model = keras.models.Model(inputs=inputs, outputs=data_layers[-1])
        
        if DEBUG:
            model.summary()

        return model



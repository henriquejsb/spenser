
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
from bindsnet.network.topology import Connection, Conv2dConnection, SparseConnection
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.utils import get_square_assignments, get_square_weights
from multiprocessing import Pool
import contextlib
from time import time as t
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        self.time = config["TRAINING"]["time"]
        self.dt = config["TRAINING"]["dt"]


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
        network = Network(learning=True)
        #input layer
        inputs = Input(n=28 * 28, shape=(1, 28, 28), traces=True)
        

        #Create layers -- ADD NEW LAYERS HERE
        layers = [(inputs,None)]
        for layer_type, layer_params in bindsnet_layers:
            #convolutional layer
            if layer_type == 'conv':

                #filters = int(layer_params['num-filters'][0])
                n_filters = 25
                #kernel_size = int(layer_params['kernel-size'][0])
                kernel_size = 16
                #padding = layer_params['padding'][0]
                padding = 0
                #strides = int(layer_params['stride'][0])
                strides = 4

                
                conv_size = int((28 - kernel_size + 2 * padding) / strides) + 1

                print("CONV N:",n_filters * conv_size * conv_size)
                conv_layer = DiehlAndCookNodes(
                    n=n_filters * conv_size * conv_size,
                    shape=(n_filters, conv_size, conv_size),
                    traces=True,
                )
                
                conv_conn = Conv2dConnection(
                    layers[-1][0],
                    conv_layer,
                    kernel_size=kernel_size,
                    stride=strides,
                    update_rule=PostPre,
                   # norm=0.4 * kernel_size**2,
                    nu=[1e-4, 1e-2],
                    wmax=1.0,
                )
                layers.append((conv_layer, conv_conn))

            elif layer_type == 'fc':
                fc_layer =  LIFNodes(
                    n=int(layer_params['num-units'][0]),
                    traces=True
                )
                fc_conn = Connection(
                    source=layers[-1][0],
                    target=fc_layer,
                    update_rule=PostPre,
                    nu=[1e-4, 1e-2],
                    wmax=1.0
                    
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
            network.add_layer(layer, name=str(ind+1))               
            network.add_connection(conn,source=last_layer, target=str(ind+1))
            last_layer = str(ind+1)
            print(last_layer)
            print(conn.update_rule)


        network.add_layer(layers[-1][0], name="Y")
        network.add_connection(layers[-1][1],source=last_layer,target="Y")
        print(network.connections)
        #print(network.connections[('X','Y')].update_rule)
        return network

    def evaluate(self, phenotype, weights_save_path, parent_weights_path,\
                num_epochs, input_size=(90, 90, 1)): #pragma: no cover
        
        model_phenotype = phenotype
        model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

        bindsnet_layers = self.get_layers(model_phenotype)
       
        print(model_phenotype)
        model = self.assemble_network(bindsnet_layers, input_size)
        
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(os.cpu_count() - 1)
        print("Running on Device = ", device)
        model.to(device)

       
        #save final moodel to file
        #model.save(weights_save_path.replace('.hdf5', '.h5'))       

        
        plot = True
        n_train = 60000
        batch_size = 32
        n_classes = 10
        n_neurons = 30
        n_updates = 50
        n_workers = 20
        update_steps = int(n_train / batch_size / n_updates)
        update_interval = update_steps * batch_size



        assignments = -torch.ones(n_neurons, device=device)
        proportions = torch.zeros((n_neurons, n_classes), device=device)
        rates = torch.zeros((n_neurons, n_classes), device=device)

        # Sequence of accuracy estimates.
        accuracy = {"all": [], "proportion": []}


         #------------------- COPIED FROM SNN EXAMPLE


        # Set up monitors for spikes and voltages
        spikes = {}
        for layer in set(model.layers):
            spikes[layer] = Monitor(
                model.layers[layer], state_vars=["s"], time=int(self.time / self.dt), device=device
            )
            model.add_monitor(spikes[layer], name="%s_spikes" % layer)

        voltages = {}
        for layer in set(model.layers) - {"X"}:
            voltages[layer] = Monitor(
                model.layers[layer], state_vars=["v"], time=int(self.time / self.dt), device=device
            )
            model.add_monitor(voltages[layer], name="%s_voltages" % layer)


        inpt_ims, inpt_axes = None, None
        spike_ims, spike_axes = None, None
        weights_im = None
        assigns_im = None
        perf_ax = None
        voltage_axes, voltage_ims = None, None


        spike_record = torch.zeros((update_interval, int(self.time / self.dt), n_neurons), device=device)

        print("Begin training.\n")
        start = t()

        #print(spikes["Y"])
        #print(spikes["Y"]["s"])

        for epoch in range(num_epochs):
            
            labels = []

            dataloader = torch.utils.data.DataLoader(
                self.dataset["train"], batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True
            )


            pbar = tqdm(total=n_train)


            for step, batch in enumerate(dataloader):
                
                # Assign labels to excitatory neurons.
                if step % update_steps == 0 and step > 0:
                    print(spike_record)
                    # Convert the array of labels into a tensor
                    label_tensor = torch.tensor(labels, device=device)

                    # Get network predictions.
                    all_activity_pred = all_activity(
                        spikes=spike_record, assignments=assignments, n_labels=n_classes
                    )
                    proportion_pred = proportion_weighting(
                        spikes=spike_record,
                        assignments=assignments,
                        proportions=proportions,
                        n_labels=n_classes,
                    )

                    # Compute network accuracy according to available classification strategies.
                    accuracy["all"].append(
                        100
                        * torch.sum(label_tensor.long() == all_activity_pred).item()
                        / len(label_tensor)
                    )
                    accuracy["proportion"].append(
                        100
                        * torch.sum(label_tensor.long() == proportion_pred).item()
                        / len(label_tensor)
                    )

                    print(
                        "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                        % (
                            accuracy["all"][-1],
                            np.mean(accuracy["all"]),
                            np.max(accuracy["all"]),
                        )
                    )
                    print(
                        "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                        " (best)\n"
                        % (
                            accuracy["proportion"][-1],
                            np.mean(accuracy["proportion"]),
                            np.max(accuracy["proportion"]),
                        )
                    )

                    # Assign labels to excitatory layer neurons.
                    assignments, proportions, rates = assign_labels(
                        spikes=spike_record,
                        labels=label_tensor,
                        n_labels=n_classes,
                        rates=rates,
                    )
                    print(assignments)

                    labels = []
                '''
                # Get next input sample.
                inputs = {"X": batch["encoded_image"]}
                print(inputs["X"].size())
                inputs = {k: v.cuda() for k, v in inputs.items()}
                print(inputs["X"].size())
                '''
                inputs = {
                    "X": batch["encoded_image"].view(int(self.time / self.dt), batch_size, 1, 28, 28).to(device)
                }
                #print(inputs["X"].size())
                inputs = {k: v.cuda() for k, v in inputs.items()}
                #print(inputs["X"].size())
                #print("--")
                #return
                # Remember labels.
                labels.extend(batch["label"].tolist())

                model.run(inputs=inputs, time=self.time, input_time_dim=1)
                

                # Add to spikes recording.
                s = spikes["Y"].get("s").permute((1, 0, 2))
                spike_record[
                    (step * batch_size)
                    % update_interval : (step * batch_size % update_interval)
                    + s.size(0)
                ] = s

                # Optionally plot various simulation information.
                if step % update_steps == 0 and step > 0 and plot:
                    

                    image = batch["image"][ 0].view(28, 28)
                    inpt = inputs["X"][:,0].view(self.time, 784).sum(0).view(28, 28)
                    lable = batch["label"][0]
                    
                    n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
                    n_layers = len(model.layers)
                    last_layer = 'X' if n_layers == 2 else str(n_layers-2)
                    input_exc_weights = model.connections[(last_layer, "Y")].w
                    last_sqrt =  int(np.ceil(np.sqrt(model.layers[last_layer].n)))
                    print("WEIGHTS:")
                    print(input_exc_weights)
                    square_weights = get_square_weights(
                        input_exc_weights.view(model.layers[last_layer].n, n_neurons), n_sqrt, last_sqrt
                    )
                    square_assignments = get_square_assignments(assignments, n_sqrt)
                    
                    
                    spikes_ = {
                        layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
                    }
                    inpt_axes, inpt_ims = plot_input(
                        image, inpt, label=lable, axes=inpt_axes, ims=inpt_ims
                    )
                    spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
                    weights_im = plot_weights(square_weights, im=weights_im)
                    assigns_im = plot_assignments(square_assignments, im=assigns_im)
                    perf_ax = plot_performance(
                        accuracy, x_scale=update_steps * batch_size, ax=perf_ax
                    )

                    plt.pause(1e-8)
                
                model.reset_state_variables()
                pbar.set_description_str("Train progress: ")
                pbar.update(batch_size)
        
            pbar.close()
        #print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, num_epochs, t() - start))
        print("Training complete.\n")
        #measure test performance
        
                
        # Sequence of accuracy estimates.
        accuracy = {"all": 0, "proportion": 0}

        # Record spikes during the simulation.
        #spike_record = torch.zeros(1, int(self.time / self.dt), n_neurons, device=device)
        
        if DEBUG:
            pass
            #print(phenotype, accuracy_test)

        #score.history['trainable_parameters'] = trainable_count
        #score.history['accuracy_test'] = accuracy_test


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


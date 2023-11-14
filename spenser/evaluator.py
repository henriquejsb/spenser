from spenser.data import load_dataset
from spenser.network import Network, assemble_network, train_network, get_fitness

from time import time as t

import torch
import snntorch

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



    def save(self, model, loss_fn, torch_layers, torch_learning, input_size, weights_save_path):
        # Save the models phenotype and weights 
        #opt_stade_dict = None
        #if optimizer.is_backprop:
        #    opt_stade_dict = optimizer.state_dict()
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            #"optimizer_state_dict": opt_stade_dict,
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

        
        
        model = Network(torch_layers,input_size)

  
        
        model.to(device)
        #print(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, loss_fn, torch_layers, torch_learning

    def testing_performance(self, weights_save_path):
        testloader = self.dataset["test"]
        num_steps = int(self.config["TRAINING"]["num_steps"])
        model, loss_fn, torch_layers, torch_learning = self.load(weights_save_path)
        #model.to(device)
        accuracy_test = get_fitness(model,testloader,num_steps)
        return accuracy_test

    def retrain_longer(self, weights_save_path, num_epochs, save_path):

        print("NOT IMPLEMENTED YET")
        exit(0)
        '''
        model, loss_fn, torch_layers, torch_learning = self.load(weights_save_path)
        #model.to(device)

        trainloader = self.dataset["evo_train"]
        testloader = self.dataset["evo_test"]
        train=self.dataset["evo_train_original"]
        test=self.dataset["evo_test_original"]
        
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
            #(model,dataset,dataloader,optimizer_genotype,loss_fn,num_epochs=0,cmaes_iterations=0,config=None):
            acc_hist, loss_hist, time_stats = train_network(model=model,
                                                            dataset = train,
                                                            dataloader = trainloader,
                                                            optimizer_genotype = torch_learning
                                                            loss_fn=loss_fn,
                                                            num_epochs=1,
                                                            cmaes_iterations=config["TRAINING"]["CMA-ES"][])
            history['accuracy'] += acc_hist
            history['loss'] += loss_hist
            history['time_stats'] += [time_stats]


            acc_hist, loss_hist, time_stats = train_network(model,testloader,optimizer,loss_fn,1)
            history['accuracy'] += acc_hist
            history['loss'] += loss_hist
            history['time_stats'] += [time_stats]

        self.save(model, optimizer, loss_fn, torch_layers, torch_learning, input_size, save_path)
        return history
        '''
    def evaluate(self, phenotype, weights_save_path, parent_weights_path,\
                num_epochs, cmaes_iterations): #pragma: no cover
        
        start = t()
        trainloader = self.dataset["evo_train"]
        testloader = self.dataset["evo_test"]
        train = self.dataset["evo_train_original"]
        
        

        loss_hist = []
        acc_hist = []
        history = {}

        num_steps = int(self.config["TRAINING"]["num_steps"])
        input_size = self.dataset["input_size"]
       

        time_stats = {}

        model = None
        optimizer = None
        loss_fn = None

       

        if num_epochs > 0:
            model_phenotype, learning_phenotype = phenotype.split('learning:')
            
            learning_phenotype = 'learning:'+learning_phenotype.rstrip().lstrip()
            model_phenotype = model_phenotype.rstrip().lstrip().replace('  ', ' ')

            torch_layers = self.get_layers(model_phenotype)
            torch_learning = self.get_learning(learning_phenotype)
        
            if DEBUG:
                print(torch_layers)
            success = assemble_network(torch_layers, input_size)
            if not success:
                if DEBUG:
                    print('INVALID INDIVIDUAL')
                return None
            
            
            model = Network(torch_layers,input_size)
            model.to(device)
            
            #get_fitness(model,testloader, num_steps)

            #TODO ADD LOSS FUNCTION AS EVOL PARAM
            loss_fn = eval(self.config["TRAINING"]["loss_fn"])

            cmaes_iterations = len(train) // self.config["TRAINING"]["batch_size"] + 1
            print("CMAES ITERATIONS",cmaes_iterations)
            #loss_fn = SF.ce_rate_loss()
            #print("Rate loss")

            model,acc_hist, loss_hist, time_stats, cmaes_logger = train_network(model=model,
                                                            dataset=train,
                                                            dataloader=trainloader,
                                                            optimizer_genotype=torch_learning,
                                                            loss_fn=loss_fn,
                                                            num_epochs=num_epochs,
                                                            cmaes_iterations=cmaes_iterations,
                                                            config=self.config)
            if DEBUG:
                print("Finished training")
            history['accuracy'] = acc_hist
            history['loss'] = loss_hist
            history['cmaes_logger'] = cmaes_logger
            
        
        else:
            #Is parent, already trained
            if DEBUG:
                print("Just loading prev")
            model, loss_fn, torch_layers, torch_learning = self.load(parent_weights_path)
            model.to(device)

        self.save(model, loss_fn, torch_layers, torch_learning, input_size, weights_save_path)
        if DEBUG:
            print("Saved model")
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





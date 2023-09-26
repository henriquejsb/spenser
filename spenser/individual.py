from spenser.module import Module
from multiprocessing import Pool
from multiprocessing import Queue
from multiprocessing import set_start_method
from multiprocessing import Process
import torch, torch.nn as nn
import traceback

DEBUG = True

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

    

    return_val, scnn_eval, phenotype, weights_save_path, parent_weights_path, num_epochs, cmaes_iterations = args
    
    try:
        history = scnn_eval.evaluate(phenotype, weights_save_path, parent_weights_path, num_epochs, cmaes_iterations)
        
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
        self.cmaes_iterations = 0
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
                            self.num_epochs, self.cmaes_iterations)])
        # run the process
        process.start()

        process.join()
        if DEBUG:
            print("Process terminated")
        metrics = return_val.get()
        

        
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

                if 'cmaes_logger' in metrics:
                    self.metrics['cmaes_logger'] = metrics['cmaes_logger']
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

        else:
            self.metrics = None
            self.fitness = -1
            self.num_epochs = 0
            self.trainable_parameters = -1
            self.current_time = 0
        
       
        return self.fitness


import os
import sys, getopt
from fast_denser.grammar import Grammar
from fast_denser.utils import Evaluator, Individual
import json
from pathlib import Path
from os import makedirs
import random
import numpy as np
from jsmin import jsmin
from copy import deepcopy
from glob import glob
import pickle
from shutil import copyfile
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from multiprocessing import set_start_method

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def save_pop(population, save_path, run, gen):
    """
        Save the current population statistics in json.
        For each individual:
            .id: unique generation identifier
            .phenotype: phenotype of the individual
            .fitness: fitness of the individual
            .metrics: other evaluation metrics (e.g., loss, accuracy)
            .trainable_parameters: number of network trainable parameters
            .num_epochs: number of performed training epochs
            .time: time (sec) the network took to perform num_epochs
            .train_time: maximum time (sec) that the network is allowed to train for



        Parameters
        ----------
        population : list
            list of Individual instances

        save_path : str
            path to the json file

        run : int
            current evolutionary run

        gen : int
            current generation

    """

    json_dump = []
    population = sorted(population, key=lambda x: x.fitness, reverse=True)
    for ind in population:
        json_dump.append({
                          'id': str(ind.id),
                          'phenotype': str(ind.phenotype),
                          'fitness': str(ind.fitness),
                          'metrics': str(ind.metrics),
                          'trainable_parameters': str(ind.trainable_parameters),
                          'num_epochs': str(ind.num_epochs),
                          'time': str(ind.time),
                          'train_time': str(ind.train_time),
                          'was_the_parent': str(ind.is_parent)})

    with open(Path('%s/run_%d/gen_%d.csv' % (save_path, run, gen)), 'w') as f_json:
        f_json.write(json.dumps(json_dump, indent=4))


def get_total_epochs(save_path, run, last_gen):
    """
        Compute the total number of performed epochs.

        Parameters
        ----------
        save_path: str
            path where the ojects needed to resume evolution are stored.

        run : int
            current evolutionary run

        last_gen : int
            count the number of performed epochs until the last_gen generation


        Returns
        -------
        total_epochs : int
            sum of the number of epochs performed by all trainings
    """

    total_epochs = 0
    for gen in range(0, last_gen+1):
        j = json.load(open(Path('%s/run_%d/gen_%d.csv' % (save_path, run, gen))))
        num_epochs = [int(elm['num_epochs']) for elm in j]
        total_epochs += sum(num_epochs)

    return total_epochs


def pickle_evaluator(evaluator, save_path, run):
    """
        Save the Evaluator instance to later enable resuming evolution

        Parameters
        ----------
        evaluator : Evaluator
            instance of the Evaluator class

        save_path: str
            path to the json file

        run : int
            current evolutionary run

    """

    with open(Path('%s/run_%d/evaluator.pkl' % (save_path, run)), 'wb') as handle:
        pickle.dump(evaluator, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_population(population, parent, best_fitness, save_path, run):
    """
        Save the objects (pickle) necessary to later resume evolution:
        Pickled objects:
            .population
            .parent
            .random states: numpy and random
        Useful for later conducting more generations.
        Replaces the objects of the previous generation.

        Parameters
        ----------
        population : list
            list of Individual instances

        parent : Individual
            fittest individual that will seed the next generation

        save_path: str
            path to the json file

        run : int
            current evolutionary run
    """

    with open(Path('%s/run_%d/population.pkl' % (save_path, run)), 'wb') as handle_pop:
        pickle.dump(population, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path('%s/run_%d/parent.pkl' % (save_path, run)), 'wb') as handle_pop:
        pickle.dump(parent, handle_pop, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path('%s/run_%d/random.pkl' % (save_path, run)), 'wb') as handle_random:
        pickle.dump(random.getstate(), handle_random, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path('%s/run_%d/numpy.pkl' % (save_path, run)), 'wb') as handle_numpy:
        pickle.dump(np.random.get_state(), handle_numpy, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path('%s/run_%d/best_fitness.pkl' % (save_path, run)), 'wb') as handle_best:
        pickle.dump(best_fitness, handle_best, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_population(save_path, run):
    """
        Save the objects (pickle) necessary to later resume evolution.
        Useful for later conducting more generations.
        Replaces the objects of the previous generation.
        Returns None in case any generation has been performed yet.


        Parameters
        ----------
        save_path: str
            path where the ojects needed to resume evolution are stored.

        run : int
            current evolutionary run


        Returns
        -------
        last_generation : int
            idx of the last performed generation

        pickle_evaluator : Evaluator
            instance of the Evaluator class used for evaluating the individuals.
            Loaded because it has the data used for training.

        pickle_population : list
            population of the last performed generation

        pickle_parent : Individual
            fittest individual of the last performed generation

        pickle_population_fitness : list
            ordered list of fitnesses of the last population of individuals

        pickle_random : tuple
            Random random state

        pickle_numpy : tuple
            Numpy random state
    """

    csvs = glob(str(Path('%s' % save_path, 'run_%d' % run, '*.csv' )))
    
    if csvs:
        csvs = [int(csv.split(os.sep)[-1].replace('gen_','').replace('.csv','')) for csv in csvs]
        last_generation = max(csvs)

        with open(Path('%s' % save_path, 'run_%d' % run, 'evaluator.pkl'), 'rb') as handle_eval:
            pickle_evaluator = pickle.load(handle_eval)

        with open(Path('%s' % save_path, 'run_%d' % run, 'population.pkl'), 'rb') as handle_pop:
            pickle_population = pickle.load(handle_pop)

        with open(Path('%s' % save_path, 'run_%d' % run, 'parent.pkl'), 'rb') as handle_parent:
            pickle_parent = pickle.load(handle_parent)
        
        with open(Path('%s' % save_path, 'run_%d' % run, 'best_fitness.pkl'), 'rb') as handle_best:
            pickle_best_fitness = pickle.load(handle_best)

        pickle_population_fitness = [ind.fitness for ind in pickle_population]

        with open(Path('%s' % save_path, 'run_%d' % run, 'random.pkl'), 'rb') as handle_random:
            pickle_random = pickle.load(handle_random)

        with open(Path('%s' % save_path, 'run_%d' % run, 'numpy.pkl'), 'rb') as handle_numpy:
            pickle_numpy = pickle.load(handle_numpy)

        total_epochs = get_total_epochs(save_path, run, last_generation)

        return last_generation, pickle_evaluator, pickle_population, pickle_parent, \
               pickle_population_fitness, pickle_random, pickle_numpy, total_epochs, pickle_best_fitness

    else:
        return None


def select_fittest(population, population_fits, grammar, cnn_eval, gen, save_path): #pragma: no cover
    """
        Select the parent to seed the next generation.


        Parameters
        ----------
        population : list
            list of instances of Individual

        population_fits : list
            ordered list of fitnesses of the population of individuals

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the genotype
            to phenotype mapping

        cnn_eval : Evaluator
            Evaluator instance used to train the networks

        datagen : keras.preprocessing.image.ImageDataGenerator
            Data augmentation method image data generator for the training data

        datagen_test : keras.preprocessing.image.ImageDataGenerator
            Data augmentation method image data generator for the validation and test data

        gen : int
            current generation of the ES

        save_path: str
            path where the ojects needed to resume evolution are stored.

        default_train_time : int
            default training time


        Returns
        -------
        parent : Individual
            individual that seeds the next generation
    """


    #Get best individual just according to fitness
    idx_max = np.argmax(population_fits)
    parent = population[idx_max]    
    return deepcopy(parent)
    '''
    #however if the parent is not the elite, and the parent is trained for longer, the elite
    #is granted the same evaluation time.
    if parent.train_time > default_train_time:
        retrain_elite = False
        if idx_max != 0 and population[0].train_time > default_train_time and population[0].train_time < parent.train_time:
            retrain_elite = True
            elite = population[0]
            elite.train_time = parent.train_time
            elite.evaluate(grammar, cnn_eval, datagen, datagen_test, '%s/best_%d_%d.hdf5' % (save_path, gen, elite.id), '%s/best_%d_%d.hdf5' % (save_path, gen, elite.id))
            population_fits[0] = elite.fitness

        min_train_time = min([ind.current_time for ind in population])

        #also retrain the best individual that is trained just for the default time
        retrain_10min = False
        if min_train_time < parent.train_time:
            ids_10min = [ind.current_time == min_train_time for ind in population]
    
            if sum(ids_10min) > 0:
                retrain_10min = True
                indvs_10min = np.array(population)[ids_10min]
                max_fitness_10min = max([ind.fitness for ind in indvs_10min])
                idx_max_10min = np.argmax(max_fitness_10min)
                parent_10min = indvs_10min[idx_max_10min]

                parent_10min.train_time = parent.train_time

                parent_10min.evaluate(grammar, cnn_eval, datagen, datagen_test, '%s/best_%d_%d.hdf5' % (save_path, gen, parent_10min.id), '%s/best_%d_%d.hdf5' % (save_path, gen, parent_10min.id))

                population_fits[population.index(parent_10min)] = parent_10min.fitness


        #select the fittest amont all retrains and the initial parent
        if retrain_elite:
            if retrain_10min:
                if parent_10min.fitness > elite.fitness and parent_10min.fitness > parent.fitness:
                    return deepcopy(parent_10min)
                elif elite.fitness > parent_10min.fitness and elite.fitness > parent.fitness:
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
            else:
                if elite.fitness > parent.fitness:
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
        elif retrain_10min:
            if parent_10min.fitness > parent.fitness:
                return deepcopy(parent_10min)
            else:
                return deepcopy(parent)
        else:
            return deepcopy(parent)

    return deepcopy(parent)
    '''


def mutation_dsge(layer, grammar):
    """
        DSGE mutations (check DSGE for futher details)


        Parameters
        ----------
        layer : dict
            layer to be mutated (DSGE genotype)

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the genotype
            to phenotype mapping
    """

    nt_keys = sorted(list(layer.keys()))
    nt_key = random.choice(nt_keys)
    nt_idx = random.randint(0, len(layer[nt_key])-1)

    sge_possibilities = []
    random_possibilities = []
    if len(grammar.grammar[nt_key]) > 1:
        sge_possibilities = list(set(range(len(grammar.grammar[nt_key]))) -\
                                 set([layer[nt_key][nt_idx]['ge']]))
        random_possibilities.append('ge')

    if layer[nt_key][nt_idx]['ga']:
        random_possibilities.extend(['ga', 'ga'])

    if random_possibilities:
        mt_type = random.choice(random_possibilities)

        if mt_type == 'ga':
            var_name = random.choice(sorted(list(layer[nt_key][nt_idx]['ga'].keys())))
            var_type, min_val, max_val, values = layer[nt_key][nt_idx]['ga'][var_name]
            value_idx = random.randint(0, len(values)-1)

            if var_type == 'int':
                new_val = random.randint(min_val, max_val)
            elif var_type == 'float':
                new_val = values[value_idx]+random.gauss(0, 0.15)
                new_val = np.clip(new_val, min_val, max_val)

            layer[nt_key][nt_idx]['ga'][var_name][-1][value_idx] = new_val

        elif mt_type == 'ge':
            layer[nt_key][nt_idx]['ge'] = random.choice(sge_possibilities)

        else:
            return NotImplementedError


def mutation(individual, grammar, add_layer, re_use_layer, remove_layer, dsge_layer, macro_layer):
    """
        Network mutations: add and remove layer, add and remove connections, macro structure


        Parameters
        ----------
        individual : Individual
            individual to be mutated

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the genotype
            to phenotype mapping

        add_layer : float
            add layer mutation rate

        re_use_layer : float
            when adding a new layer, defines the mutation rate of using an already
            existing layer, i.e., copy by reference

        remove_layer : float
            remove layer mutation rate

        add_connection : float
            add connection mutation rate

        remove_connection : float
            remove connection mutation rate

        dsge_layer : float
            inner lever genotype mutation rate

        macro_layer : float
            inner level of the macro layers (i.e., learning, data-augmentation) mutation rate

        train_longer : float
            increase the training time mutation rate

        default_train_time : int
            default training time

        Returns
        -------
        ind : Individual
            mutated individual
    """

    #copy so that elite is preserved
    ind = deepcopy(individual)



    #in case the individual is mutated in any of the structural parameters
    #the training time is reseted
    ind.current_time = 0
    ind.num_epochs = 0
    ind.is_parent = False
    ind.metrics = None
    for module in ind.modules:

        #add-layer (duplicate or new)
        for _ in range(random.randint(1,2)):
            if len(module.layers) < module.max_expansions and random.random() <= add_layer:
                if random.random() <= re_use_layer:
                    new_layer = random.choice(module.layers)
                else:
                    new_layer = grammar.initialise(module.module)

                insert_pos = random.randint(0, len(module.layers))

                
                module.layers.insert(insert_pos, new_layer)

             

        #remove-layer
        for _ in range(random.randint(1,2)):
            if len(module.layers) > module.min_expansions and random.random() <= remove_layer:
                remove_idx = random.randint(0, len(module.layers)-1)
                del module.layers[remove_idx]
                
                


        for layer_idx, layer in enumerate(module.layers):
            #dsge mutation
            if random.random() <= dsge_layer:
                mutation_dsge(layer, grammar)

           


    #macro level mutation
    for macro_idx, macro in enumerate(ind.macro): 
        if random.random() <= macro_layer:
            mutation_dsge(macro, grammar)
                    

    return ind


def load_config(config_file):
    """
        Load configuration json file.


        Parameters
        ----------
        config_file : str
            path to the configuration file
            
        Returns
        -------
        config : dict
            configuration json file
    """

    with open(Path(config_file)) as js_file:
        minified = jsmin(js_file.read())

    config = json.loads(minified)
    #config["TRAINING"]["fitness_metric"] = eval(config["TRAINING"]["fitness_metric"])
    #config["DATASET"]["shape"] = eval(config["DATASET"]["shape"])
    return config


def main(run, dataset, config_file, grammar_path, evaluate_test, retrain_epochs): #pragma: no cover
    """
        (1+lambda)-ES


        Parameters
        ----------
        run : int
            evolutionary run to perform

        dataset : str
            dataset to be solved

        config_file : str
            path to the configuration file

        grammar_path : str
            path to the grammar file
    """ 
    #load config file
    config = load_config(config_file)
    skip_parent = eval(config["EVOLUTIONARY"]["skip_parent"])
    #load grammar
    grammar = Grammar(grammar_path)

    #best fitness so far
    best_fitness = None

    #load previous population content (if any)
    unpickle = unpickle_population(config["EVOLUTIONARY"]["save_path"], run)

    set_start_method('spawn')
     #if there is not a previous population
    if unpickle is None:
        makedirs('%s/run_%d/' % (config["EVOLUTIONARY"]["save_path"], run), exist_ok=True)

        #set random seeds
        random.seed(config["EVOLUTIONARY"]["random_seeds"][run])
        np.random.seed(config["EVOLUTIONARY"]["numpy_seeds"][run])

        #create evaluator
        cnn_eval = Evaluator(dataset, config)

        #save evaluator
        pickle_evaluator(cnn_eval, config["EVOLUTIONARY"]["save_path"], run)

        #status variables
        last_gen = -1
        total_epochs = 0

    #in case there is a previous population, load it
    else:
        last_gen, cnn_eval, population, parent, population_fits, pkl_random, pkl_numpy, total_epochs,best_fitness = unpickle
        random.setstate(pkl_random)
        np.random.set_state(pkl_numpy)

    if evaluate_test or retrain_epochs:
        best_path = str(Path('%s' % config["EVOLUTIONARY"]["save_path"], 
                             'run_%d' % run, 'best'))
        accuracy_log_file = str(Path('%s' % config["EVOLUTIONARY"]["save_path"], 
                             'run_%d' % run, 'best_test_accuracy.txt'))
        if retrain_epochs:
            retrain_path = str(Path('%s' % config["EVOLUTIONARY"]["save_path"], 
                             'run_%d' % run, 'best_retrained_%d' % retrain_epochs))
            retrain_log_file = str(Path('%s' % config["EVOLUTIONARY"]["save_path"], 
                             'run_%d' % run, 'best_retrained_%d_log.txt' % retrain_epochs))
            history = cnn_eval.retrain_longer(best_path, retrain_epochs, retrain_path)
            with open(retrain_log_file, 'w') as f:
                f.write(str(history)+'\n')
            
            
            best_path = retrain_path
        
        
        test_accuracy = cnn_eval.testing_performance(best_path)
        print("Test accuracy:",test_accuracy)
        with open(accuracy_log_file, 'w') as f:
            f.write(str(test_accuracy))
        return

    for gen in range(last_gen+1, config["EVOLUTIONARY"]["num_generations"]):

        #check the total number of epochs (stop criteria)
        #if total_epochs is not None and total_epochs >= config["EVOLUTIONARY"]["max_epochs"]:
        #    break
        if gen == 0:
            print('[%d] Creating the initial population' % (run))
            print('[%d] Performing generation: %d' % (run, gen))
            population = [Individual(config["NETWORK"]["network_structure"],config["NETWORK"]["macro_structure"],\
            config["NETWORK"]["output"],\
                                     
                                _id_).initialise(grammar, config["EVOLUTIONARY"]["MUTATIONS"]["reuse_layer"], config["NETWORK"]["network_structure_init"]) \
                                for _id_ in range(config["EVOLUTIONARY"]["lambda"])]
            
            population_fits = []
            for idx, ind in enumerate(population):
                ind.current_time = 0
                ind.num_epochs = int(config["TRAINING"]["epochs"])
                population_fits.append(ind.evaluate(grammar, cnn_eval,'%s/run_%d/best_%d_%d' % (config["EVOLUTIONARY"]["save_path"], run, gen, idx)))
                ind.id = idx

        else:
            print('[%d] Performing generation: %d' % (run, gen))
            
            #generate offspring (by mutation)
            offspring = [mutation(parent, grammar, config["EVOLUTIONARY"]["MUTATIONS"]["add_layer"],
                                  config["EVOLUTIONARY"]["MUTATIONS"]["reuse_layer"], config["EVOLUTIONARY"]["MUTATIONS"]["remove_layer"],
                                  config["EVOLUTIONARY"]["MUTATIONS"]["dsge_layer"], config["EVOLUTIONARY"]["MUTATIONS"]["macro_layer"]) 
                                  for _ in range(config["EVOLUTIONARY"]["lambda"])]

            population = [parent] + offspring

            #set elite variables to re-evaluation
            #population[0].current_time = 0
            
            
            population[0].num_epochs = 0
            population[0].is_parent = True
            parent_id = parent.id
            parent.is_parent = True
            #evaluate population
            population_fits = []
            for idx, ind in enumerate(population):
                if ind.is_parent and skip_parent:
                    #population_fits.append(ind.fitness)
                    #continue
                    #print("Is parent")
                    ind.num_epochs = 0
                else:
                    #print("Not parent")
                    ind.num_epochs = int(config["TRAINING"]["epochs"])
                
                #input()
                population_fits.append(ind.evaluate(grammar, cnn_eval,  '%s/run_%d/best_%d_%d' % (config["EVOLUTIONARY"]["save_path"], run, gen, idx), '%s/run_%d/best_%d_%d' % (config["EVOLUTIONARY"]["save_path"], run, gen-1, parent_id)))
                ind.id = idx

        #select parent
        parent = select_fittest(population, population_fits, grammar, cnn_eval,\
                                 gen, \
                                config["EVOLUTIONARY"]["save_path"]+'/run_'+str(run))

        #remove temporary files to free disk space
        if gen > 1:
            for x in range(len(population)):
                if os.path.isfile(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.checkpoint' % (gen-2, x))):
                    os.remove(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.checkpoint' % (gen-2, x)))
                if os.path.isfile(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.phenotype' % (gen-2, x))):    
                    os.remove(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.phenotype' % (gen-2, x)))

        #update best individual
        if best_fitness is None or parent.fitness > best_fitness:
            best_fitness = parent.fitness

            if os.path.isfile(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.checkpoint' % (gen, parent.id))):
                copyfile(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.checkpoint' % (gen, parent.id)), Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best.checkpoint'))
            #if os.path.isfile(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.phenotype' % (gen, parent.id))):
            #    copyfile(Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best_%d_%d.phenotype' % (gen, parent.id)), Path('%s' % config["EVOLUTIONARY"]["save_path"], 'run_%d' % run, 'best.phenotype'))
            
            with open('%s/run_%d/best_parent.pkl' % (config["EVOLUTIONARY"]["save_path"], run), 'wb') as handle:
                pickle.dump(parent, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('[%d] Best fitness of generation %d: %f' % (run, gen, max(population_fits)))
        print('[%d] Best overall fitness: %f' % (run, best_fitness))

        #save population
        save_pop(population, config["EVOLUTIONARY"]["save_path"], run, gen)
        pickle_population(population, parent, best_fitness, config["EVOLUTIONARY"]["save_path"], run)

        total_epochs += sum([ind.num_epochs for ind in population])

    return

def process_input(argv): #pragma: no cover
    """
        Maps and checks the input parameters and call the main function.

        Parameters
        ----------
        argv : list
            argv from system
    """

    dataset = None
    config_file = None
    run = 0
    grammar = None
    evaluate_test = False
    retrain_epochs = 0
    try:
        opts, args = getopt.getopt(argv, "hd:c:r:g:R:E",["dataset=","config=","run=","grammar=","evaluate","retrain="]   )
    except getopt.GetoptError:
        print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>')
        sys.exit(2)


    for opt, arg in opts:
        if opt == '-h':
            print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammra>')
            sys.exit()

        elif opt in ("-d", "--dataset"):
            dataset = arg

        elif opt in ("-c", "--config"):
            config_file = arg

        elif opt in ("-r", "--run"):
            run = int(arg)

        elif opt in ("-g", "--grammar"):
            grammar = arg
        
        elif opt in ("--retrain"):
            retrain_epochs = int(arg)
        
        elif opt in ("--evaluate"):
            evaluate_test = True
            


    error = False

    #check if mandatory variables are all set
    if dataset is None:
        print('The dataset (-d) parameter is mandatory.')
        error = True

    if config_file is None:
        print('The config. file parameter (-c) is mandatory.')
        error = True

    if grammar is None:
        print('The grammar (-g) parameter is mandatory.')
        error = True

    if error:
        print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>')
        exit(-1)

    #check if files exist
    if not os.path.isfile(grammar):
        print('Grammar file does not exist.')
        error = True

    if not os.path.isfile(config_file):
        print('Configuration file does not exist.')
        error = True


    if not error:
        main(run, dataset, config_file, grammar, evaluate_test, retrain_epochs)
    else:
        print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>')



if __name__ == '__main__': #pragma: no cover
    
    process_input(sys.argv[1:]) 

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


def main(run, dataset, config_file, grammar_path): #pragma: no cover
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

    #load grammar
    grammar = Grammar(grammar_path)

    #best fitness so far
    best_fitness = None

    makedirs('%s/run_%d/' % (config["EVOLUTIONARY"]["save_path"], run), exist_ok=True)

    #set random seeds
    random.seed(config["EVOLUTIONARY"]["random_seeds"][run])
    np.random.seed(config["EVOLUTIONARY"]["numpy_seeds"][run])

    #create evaluator
    scnn_eval = Evaluator(dataset, config)

    #status variables
    last_gen = -1
    total_epochs = 0
    for gen in range(last_gen+1, config["EVOLUTIONARY"]["num_generations"]):

        #check the total number of epochs (stop criteria)
        if total_epochs is not None and total_epochs >= config["EVOLUTIONARY"]["max_epochs"]:
            break
        if gen == 0:
            print('[%d] Creating the initial population' % (run))
            print('[%d] Performing generation: %d' % (run, gen))
            population = [Individual(config["NETWORK"]["network_structure"],\
                                _id_).initialise(grammar, config["EVOLUTIONARY"]["MUTATIONS"]["reuse_layer"], config["NETWORK"]["network_structure_init"]) \
                                for _id_ in range(config["EVOLUTIONARY"]["lambda"])]
            
            population_fits = []
            for idx, ind in enumerate(population):
                ind.current_time = 0
                ind.num_epochs = 0
                population_fits.append(ind.evaluate(grammar, scnn_eval,'%s/run_%d/best_%d_%d.hdf5' % (config["EVOLUTIONARY"]["save_path"], run, gen, idx)))
                ind.id = idx
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

    try:
        opts, args = getopt.getopt(argv, "hd:c:r:g:",["dataset=","config=","run=","grammar="]   )
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
        main(run, dataset, config_file, grammar)
    else:
        print('f_denser.py -d <dataset> -c <config> -r <run> -g <grammar>')



if __name__ == '__main__': #pragma: no cover
    
    process_input(sys.argv[1:]) 
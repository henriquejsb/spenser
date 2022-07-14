import os
import sys, getopt
from fast_denser.utils import load_config
from fast_denser.grammar import Grammar
import json
from pathlib import Path
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
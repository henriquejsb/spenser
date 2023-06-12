import random 

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

        layers : list
            list of layers of the module

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
        """

        num_expansions = random.choice(init_max[self.module])

        #Initialise layers
        for idx in range(num_expansions):
            if idx>0 and random.random() <= reuse:
                r_idx = random.randint(0, idx-1)
                self.layers.append(self.layers[r_idx])
            else:
                self.layers.append(grammar.initialise(self.module))

#!/usr/bin/env python

###############################################################################
# PARAMETERS:
# argv[1] is the candidate configuration number
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the function name
# argv[5] is the optimizer name
# The rest (argv[6:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################
import datetime
import os
import os.path
import re
import subprocess
import sys

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, "w")
blockPrint()
# sys.path.append('./nevergrad')
# import nevergrad as ng
from nevergrad.functions import ArtificialFunction
from nevergrad.benchmark.xpbase import Experiment
from nevergrad.benchmark import experiments
from nevergrad.optimization.base import ConfiguredOptimizer
from nevergrad.optimization.optimizerlib import *
import argparse



def enablePrint():
    sys.stdout = sys.__stdout__

# Useful function to print errors.
def target_runner_error(msg:str = None) -> None:
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)
    
    
def get_function(name:str = None, 
                        seed:int = None, 
                        block_dimension:int = None, 
                        rotation:bool = False, 
                        num_blocks:int = 1, 
                        noise_level:float = 0) -> ArtificialFunction:
    """Create artificial function

    Args:
        name (str, optional): Name of the predefiened functions. Defaults to None.
        seed (int, optional): Random seed. Defaults to None.
        block_dimension (int, optional): Block dimension. Defaults to None.
        rotation (bool, optional): Rotation. Defaults to False.
        num_blocks (int, optional): Num blocks. Defaults to 1.
        noise_level (float, optional): Noise level. Defaults to 0.

    Returns:
        ArtificialFunction: Nevergrad implementation of artificial function
    """
    artif_funct = ArtificialFunction(name=name, block_dimension=block_dimension, rotation=rotation, num_blocks=num_blocks, noise_level=noise_level)
    return artif_funct


# Nevergrad optitimizers 
    # __all__ = [
    #     "ParametrizedOnePlusOne",
    #     "ParametrizedCMA", <-
    #     "ParametrizedBO",
    #     "DifferentialEvolution",
    #     "EvolutionStrategy",
    #     "ScipyOptimizer",
    #     "Pymoo",
    #     "RandomSearchMaker",
    #     "SamplingSearch",
    #     "Chaining",
    #     "EMNA", <-
    #     "ParametrizedTBPSA",
    #     "NoisySplit",
    #     "ConfSplitOptimizer",
    #     "ConfPortfolio",
    #     "BayesOptim",
    # ]

# TODO: popsize coresponding with num_workers
# TODO: Fix SEED
def get_optimizer(name:str = None, cand_params:list = None) -> ConfiguredOptimizer:
    """Create optimizer for a given name

    Args:
        name (str, optional): Name of the optimizer. Defaults to None.

    Returns:
        ConfiguredOptimizer: Nevergrad implementation of optimizer.
    """
    if name == 'CMA':
        diagonal_flag = False
        while cand_params:
            param = cand_params.pop(0)
            value = cand_params.pop(0)
            if param == "--scale":
                scale = float(value)
            elif param == "--popsize":
                popsize = int(value)
            elif param == "--elitist":
                elitist = True if value == 'True' else False
            elif param == "--diagonal":
                diagonal = True if value == 'True' else False
                diagonal_flag = True
            elif diagonal_flag and param == "--fcmaes":
                fcmaes = True if value == 'True' else False
            elif param == "--random_init":
                random_init = True if value == 'True' else False
            else:
                target_runner_error("unknown parameter %s" % (param))

        return ParametrizedCMA(scale=scale, popsize=popsize, elitist=elitist, diagonal=diagonal, random_init=random_init).set_name('CMA_my', register=True)
    else:
        return None       


def get_experiment(optimizer:ConfiguredOptimizer, artificial_function:ArtificialFunction, budged:int = 1, seed:int = None) -> Experiment:
    experiment = Experiment(function=artificial_function, optimizer=optimizer, budget=budged, seed=seed)
    experiment._run_with_error()
    enablePrint()
    print(experiment.result.get('loss'))
    sys.exit(0)
    


if __name__=='__main__':
    
    if len(sys.argv) < 5:
        print("\nUsage: ./target-runner.py <configuration_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
        sys.exit(1)

    # Get the parameters as command line arguments.
    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    benchmark = sys.argv[4]
    cand_params = sys.argv[5:]

    # Default values (if any)
    name = None
    rotation = None
    d = None
    budget = None
    scale = None
    popsize = None
    elitist = None
    diagonal = False
    fcmaes = None
    random_init = None
    # Parse parameters
    parameters_optimizer = []
    
    while cand_params:
        # Get and remove first and second elements.
        param_value = cand_params.pop(0)
        param = param_value.split('=')[0]
        value = param_value.split("=")[1]
        if param == "--name":
            name = str(value)
        elif param == "--rotation":
            rotation = True if value == 'True' else False
        elif param == "--budget":
            budget = int(value)
        elif param == "--d":
            d = int(value)
        else:
            parameters_optimizer.append(param)
            parameters_optimizer.append(value)
    
    parameters_function = {
        'name': name,
        'rotation': rotation,
        'd': d
    }
    
    
    
    method = getattr(experiments, benchmark)
    artificial_function = method(options=parameters_function)
    optimizer = get_optimizer('CMA', cand_params=parameters_optimizer)
    
    get_experiment(optimizer=optimizer, artificial_function=artificial_function, budged=budget)
            

    
# seed = 7
# function_name = 'hm'
# block_dimension = 100
# num_blocks = 1
# # pop size -> number of workers
# artificial_function = get_function(function_name, seed, block_dimension)
# optimizer = get_optimizer("CMA")
# get_experiment(optimizer, artificial_function)

    
    
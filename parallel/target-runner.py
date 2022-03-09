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
print("Program start")
import datetime
import os
import os.path
import re
import subprocess
import sys
from abc import ABC
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, "w")
# blockPrint()
from nevergrad.functions import ArtificialFunction
from nevergrad.benchmark.xpbase import Experiment
from nevergrad.benchmark import experiments
from nevergrad.optimization.base import ConfiguredOptimizer
from nevergrad.optimization.optimizerlib import *
from nevergrad.optimization.recastlib import *
import argparse


def enablePrint():
    sys.stdout = sys.__stdout__


class ExperimentWrapper(ABC):
    def __init__(self, function_parameters:dict, budget:int, benchmark:str, optimizer_name:str, seed:int) -> None:
        super().__init__()
        self.rotation = function_parameters['rotation']
        self.dimension = function_parameters['d']
        self.function_name = function_parameters['name']
        self.benchmark = benchmark
        self.budget = budget
        self.name = optimizer_name
        self.seed = seed
        self.function_parameters = function_parameters
        
    def set_function(self):
        method = getattr(experiments, self.benchmark)
        self.artificial_function = method(options=self.function_parameters)
        
    def set_optimizer(self, cand_params:list) -> ConfiguredOptimizer:
        """Create optimizer for a given name

        Args:
            cand_params (list): List of candidate parameters for optimizer.

        Returns:
            ConfiguredOptimizer: Nevergrad implementation of optimizer.
        """
        name = self.name
        if name == 'CMA-ES':
            diagonal_flag = True
            fcmaes = False
            while cand_params:
                param = cand_params.pop(0)
                value = cand_params.pop(0)
                if param == "--scale":
                    scale = float(value)
                elif param == "--popsize_factor":
                    popsize = int(4 + int(value) * np.log(self.dimension))
                elif param == "--elitist":
                    elitist = True if value == 'True' else False
                elif param == "--diagonal":
                    diagonal = True if value == 'True' else False
                elif param == "--fcmaes":
                    print(self.dimension)
                    if self.dimension < 50:
                        fcmaes = False
                    else:
                        fcmaes = True if value == 'True' else False
                    print(fcmaes)
                elif param == "--random_init":
                    random_init = True if value == 'True' else False
                else:
                    target_runner_error("unknown parameter %s" % (param))
            if diagonal_flag and fcmaes:
                optimizer = ParametrizedCMA(scale=scale, popsize=popsize, elitist=elitist, diagonal=diagonal, random_init=random_init, fcmaes=fcmaes).set_name('CMA_my', register=True)
            else:
                optimizer = ParametrizedCMA(scale=scale, popsize=popsize, elitist=elitist, diagonal=diagonal, random_init=random_init).set_name('CMA_my', register=True)
        elif name == 'OnePlusPne':
            while cand_params:
                param = cand_params.pop(0)
                value = cand_params.pop(0)
                if param == "--noise_handling":
                    noise = str(value)
                elif param == "--mutation":
                    mutation = str(value)
                elif param == "--crossover":
                    crossover = True if value == 'True' else False
                elif param == "--use_pareto":
                    use_pareto = True if value == 'True' else False
                else:
                    target_runner_error("unknown parameter %s" % (param))

            return ParametrizedOnePlusOne(noise_handling=noise, mutation=mutation, crossover=crossover, rotation=self.rotation, use_pareto=use_pareto).set_name('OnePlusOne_my', register=True)
        elif name == "SLSQP":
            while cand_params:
                param = cand_params.pop(0)
                value = cand_params.pop(0)
                if param == "--method":
                    method = str(value)
                elif param == "--random_restart":
                    random_restart = True if value == 'True' else False
                else:
                    target_runner_error("unknown parameter %s" % (param))

            optimizer = ScipyOptimizer(method=method, random_restart=random_restart).set_name("Scipy_my", register=True)
        elif name == "Nelder-Mead":
            while cand_params:
                param = cand_params.pop(0)
                value = cand_params.pop(0)
                if param == "--method":
                    method = str(value)
                elif param == "--random_restart":
                    random_restart = True if value == 'True' else False
                else:
                    target_runner_error("unknown parameter %s" % (param))

            optimizer = ScipyOptimizer(method=method, random_restart=random_restart).set_name("Scipy_my", register=True)
        elif name == "COBYLA":
            while cand_params:
                param = cand_params.pop(0)
                value = cand_params.pop(0)
                if param == "--method":
                    method = str(value)
                elif param == "--random_restart":
                    random_restart = True if value == 'True' else False
                else:
                    target_runner_error("unknown parameter %s" % (param))

            optimizer = ScipyOptimizer(method=method, random_restart=random_restart).set_name("Scipy_my", register=True)
        elif name == "L-BFGS-B":
            while cand_params:
                param = cand_params.pop(0)
                value = cand_params.pop(0)
                if param == "--method":
                    method = str(value)
                elif param == "--random_restart":
                    random_restart = True if value == 'True' else False
                else:
                    target_runner_error("unknown parameter %s" % (param))

            optimizer = ScipyOptimizer(method=method, random_restart=random_restart).set_name("Scipy_my", register=True)
        elif name == "MetaModel":
            while cand_params:
                param = cand_params.pop(0)
                value = cand_params.pop(0)
                if param == "--multivariate_optimizer":
                    optimizer_class = eval(str(value))
                else:
                    target_runner_error("unknown parameter %s" % (param))

            optimizer = ParametrizedMetaModel(multivariate_optimizer=optimizer_class).set_name("MetaModel_my", register=True)
        elif name == "TBPSA":
            while cand_params:
                param = cand_params.pop(0)
                value = cand_params.pop(0)
                if param == "--naive":
                    naive = True if value == 'True' else False
                elif param == "--initial_popsize_factor":
                    initial_popsize = 4 * self.dimension + int(value)
                else:
                    target_runner_error("unknown parameter %s" % (param))

            optimizer = ParametrizedTBPSA(naive=naive, initial_popsize=initial_popsize).set_name("TBPSA_my", register=True)
        elif name == "Chaining":
            while cand_params:
                param = cand_params.pop(0)
                value = cand_params.pop(0)
                if param == "--optimizers":
                    optimizers = str(value).split("_")
                    optimizer_1 = eval(str(optimizers[0]))
                    optimizer_2 = eval(str(optimizers[1]))
                elif param == "--budget_chaining":
                    budget_chaining = str(value)
                else:
                    target_runner_error("unknown parameter %s" % (param))
                    
            print("In chaining works")

            optimizer = Chaining(optimizers=[optimizer_1, optimizer_2], budgets=[budget_chaining]).set_name("Chaining_my", register=True)
            
            print("Optimizer created")
        else:
            return None  
        
        self.optimizer = optimizer
        return optimizer    
    
    def get_experiment(self) -> Experiment:
        experiment = Experiment(function=self.artificial_function, optimizer=self.optimizer, budget=self.budget, seed=self.seed)
        experiment._run_with_error()
        enablePrint()
        # print(f'D={self.dimension}\nBud={self.budget}Optim={self.optimizer.name}\nFunct={self.artificial_function.name}')

        print(experiment.result.get('loss'))
        sys.exit(0)
    
# Useful function to print errors.
def target_runner_error(msg:str = None) -> None:
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)
    
# This is not used    
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


def get_optimizer(name:str = None, cand_params:list = None) -> ConfiguredOptimizer:
    """Create optimizer for a given name

    Args:
        name (str, optional): Name of the optimizer. Defaults to None.

    Returns:
        ConfiguredOptimizer: Nevergrad implementation of optimizer.
    """
    if name == 'CMA':
        # diagonal_flag = False
        while cand_params:
            param = cand_params.pop(0)
            value = cand_params.pop(0)
            if param == "--scale":
                scale = float(value)
            elif param == "--popsize_factor":
                pass
                # popsize = int(4 + int(value) * np.log(dimension))
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

        # return ParametrizedCMA(scale=scale, popsize=popsize, elitist=elitist, diagonal=diagonal, random_init=random_init).set_name('CMA_my', register=True)
    elif name == 'OnePlusPne':
        return None
        # while cand_params:
        #     param = cand_params.pop(0)
        #     value = cand_params.pop(0)
        #     if param == "--noise_handling":
        #         noise = str(value)
        #     elif param == "--popsize":
        #         popsize = int(value)
        #     elif param == "--elitist":
        #         elitist = True if value == 'True' else False
        #     elif param == "--diagonal":
        #         diagonal = True if value == 'True' else False
        #         diagonal_flag = True
        #     elif diagonal_flag and param == "--fcmaes":
        #         fcmaes = True if value == 'True' else False
        #     elif param == "--random_init":
        #         random_init = True if value == 'True' else False
        #     else:
        #         target_runner_error("unknown parameter %s" % (param))

        # return ParametrizedCMA(scale=scale, popsize=popsize, elitist=elitist, diagonal=diagonal, random_init=random_init).set_name('CMA_my', register=True)
    
    elif name == "Scipy":
        return None
    elif name == "MetaModel":
        return None
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
    seed = int(seed)
    benchmark = sys.argv[4]
    cand_params = sys.argv[5:]
    
    

    # Default values (if any)
    name = None
    rotation = None
    d = None
    budget = None
    solver = None
    
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
        elif param == "--solver":
            solver = str(value)
        else:
            parameters_optimizer.append(param)
            parameters_optimizer.append(value)
    
    parameters_function = {
        'name': name,
        'rotation': rotation,
        'd': d
    }
    
    print("Works")

    exp_wrap = ExperimentWrapper(function_parameters=parameters_function, budget=budget, benchmark=benchmark, optimizer_name=solver, seed=seed)
    # print(exp_wrap.name)
    exp_wrap.set_function()
    exp_wrap.set_optimizer(cand_params=parameters_optimizer)
    exp_wrap.get_experiment()
    # get_experiment(optimizer=optimizer, artificial_function=artificial_function, budged=budget)
            

    
# seed = 7
# function_name = 'hm'
# block_dimension = 100
# num_blocks = 1
# # pop size -> number of workers
# artificial_function = get_function(function_name, seed, block_dimension)
# optimizer = get_optimizer("CMA")
# get_experiment(optimizer, artificial_function)

    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:03:20 2018

@author: au560049
"""

"""Entry point to evolving the neural network. Start here."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
from optimizer import Optimizer
from tqdm import tqdm
import time
#import multiprocessing
import math
import sys

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.rtf'
)

def train_networks(networks, dataset):
    """Train each network.
    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    
    
    print('training each network')
    pbar = tqdm(total=len(networks))
    for network in networks:
        print('training network - inside loop')
        network.train(dataset)
        pbar.update(1)
    pbar.close()
    print('done training')
    
    
    """
    for network in networks:
        p = multiprocessing.Process(target=network.train,args=(dataset,))
        p.start()
        p.join()
    """
    
    

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.
    Args:
        networks (list): List of networks
    Returns:
        float: The average accuracy of a population of networks.
    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy
    
    print('Total accuracy : ',total_accuracy)
    print('avg accuracy : ',total_accuracy / len(networks))
    return total_accuracy / len(networks)

def get_standard_deviation(networks,avg):
    """Get the standard deviation for a group of networks.
    Args:
        networks (list): List of networks
    Returns:
        float: Standard deviation of a population of networks.
    """
    
    diffsquared = 0
    sum_diffsquared = 0
    for network in networks:
        diffsquared = (network.accuracy-avg)**2
        sum_diffsquared = diffsquared + sum_diffsquared
        
    stddev = math.sqrt((sum_diffsquared)/(len(networks)))
    print('standard deviation : ',stddev)
    return stddev

def generate(generations, population, nn_param_choices, dataset):
    """Generate a network with the genetic algorithm.
    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating
    """
    print('calling the optimizer function')
    optimizer = Optimizer(nn_param_choices)
    print('creatin NNs')
    networks = optimizer.create_population(population)
    print('done creating NNs')

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        print('sending networks to train')
        train_networks(networks, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)
        std_dev = get_standard_deviation(networks,average_accuracy)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info("Generation standard deviation: %.4f%%" % (std_dev))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.
    Args:
        networks (list): The population of networks
    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    """Evolve a network."""
    generations = 6  # Number of times to evole the population.
    population = 15  # Number of networks in each generation.
    dataset = 'sweeps'
    """
    nn_param_choices = {
        'nb_conv' : [2,3,4,5,6], ### number of convolution layers
        'conv_filters': [32,64,128,192,256,320],###number of convolutions
        'filter_size' : [2,3,4,5,6,7], ###size of each kernel,
        'nb_neurons': [64, 128, 256, 512], ###number of neurons in FC layer
        'nb_layers': [1, 2, 3, 4,5], ###Total number of FC layers
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'], ###activation function (Expectation:relu)
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', #### optimizer (Expectation: adam or variant of adam)
                      'adadelta', 'adamax', 'nadam'],
        'dropout_conv' : [0.1,0.2,0.3,0.4], ###dropout rate for a layer,
        'dropout' : [0.1,0.2,0.3,0.4]
        #'l2_lambda':[0.0001,0.001,0.01,0.1] ###l2 regularization rate
    }
    """
    
    nn_param_choices = {
        'nb_conv' : [2,3,4,5], ### number of convolution layers
        'conv_filters': [32,50,64],###number of convolutions
        'filter_size' : [2,3,4,5,6,7], ###size of each kernel,
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'], ###activation function (Expectation:relu)
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', #### optimizer (Expectation: adam or variant of adam)
                      'adadelta', 'adamax', 'nadam'],
        'dropout_conv' : [0.1,0.2,0.3,0.4], ###dropout rate for a layer,
        #'l2_lambda':[0.0001,0.001,0.01,0.1] ###l2 regularization rate
    }
    
    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))
    
    print('generating generations')
    generate(generations, population, nn_param_choices, dataset)
    
    #p = multiprocessing.Process(target=generate,args=(generations, population, nn_param_choices, dataset))
    #p.start()
    #p.join()

if __name__ == '__main__':
    
    
    #type_data = sys.argv[1:]
    
    start_time = time.clock()
    
    file = open('/home/noor/popGen/sweeps/trainingSets/final/time_file.rtf','w') 
    main()
    time_elapsed = time.clock() - start_time
    file.write(str(time_elapsed))
    file.close()
    

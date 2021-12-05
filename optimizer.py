#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:51:27 2018

@author: au560049
"""

"""

Genetic algorithm Optimization

"""

#from functools import reduce
import functools
from operator import add
import random
from network import Network

class Optimizer():
    """Class that implements genetic algorithm for soptimization."""
    
    def __init__(self, nn_param_choices, retain=0.4,
                 random_select=0.1, mutate_chance=0.3):
        """Create an optimizer.
        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated
        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices
        
    def create_population(self, count):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        Returns:
            (list): Population of network objects
        """
        print('beginning of create_population function')
        pop = []
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.nn_param_choices)
            network.create_random()

            # Add the network to our population.
            pop.append(network)
        
        print('just about to return')
        return pop
    
    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function.
        Args:
            network (network object)
        Returns:
            (float): network accuracy
        """
        return network.accuracy
    
    def grade(self, pop):
        """Find average fitness for a population.
        Args:
            pop (list): The population of networks
        Returns:
            average (float): The average accuracy of the population
        """
        summed = functools.reduce(add, (self.fitness(network) for network in pop))
        
        
        return summed / float((len(pop)))
    
    def breed(self, mother, father):
        """Make two children as parts of their parents.
        Args:
            mother (dict): Network parameters
            father (dict): Network parameters
        Returns:
            children (list): Two network objects
        """
        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the kid.
            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children
    
    def mutate(self, network):
        """Randomly mutate one part of the network.
        Args:
            network (dict): The network parameters to mutate
        Returns:
            network (Network): A randomly mutated network object
        """
        # Choose a random key.
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # Mutate one of the params.
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])

        return network
    
    def evolve(self, pop):
        """Evolve a population of networks.s
        Args:
            pop (list): A list of network parameters
        Returns:
            parents (list): The evolved population of networks
        """
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the basis scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number I want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)

        # The parents are every network I want to keep.
        parents = graded[:retain_length]

        # Randomly keep some from those I don't keep. This might produce some variation in the population
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Finding out how many spots are left to fill i.e. how many children need to be produced.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two networks retained for next population.
        while len(children) < desired_length:

            # Get a random mama and baba.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Making sure that same netwok is not picked twice
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Making sure that population doesn't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
    
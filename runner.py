import configtools
import copy
import environments
import evolver
import functools
import genome
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pyrosim
import random
import treenome


RECORD_EVAL_TIME = 50

def create_genome(config):
    # creates a random genome
    return genome.Genome.from_config(config)

def eval_batch_on_env(genomes, env, config, fitness_function):
    # evaluates batch of genomes on a single environment
    tree_kwargs = configtools.tree_from_config(config)
    sim_kwargs = configtools.sim_from_config(config)
    envs_kwargs = configtools.envs_from_config(config)

    sims = [0] * len(genomes)
    tree = treenome.Tree(**tree_kwargs)

    batch_fitness = np.zeros(len(genomes))

    for i, genome in enumerate(genomes):
        sims[i] = pyrosim.Simulator(play_blind=True, **sim_kwargs)
        genome.send_to_simulator(sims[i], tree)
        environments.send_to_simulator(sims[i], env, tree, **envs_kwargs)

        sims[i].start()

    for i in range(len(genomes)):
        sensor_data = sims[i].wait_to_finish()
        batch_fitness[i] = fitness_function(env, sensor_data, RECORD_EVAL_TIME)

    return batch_fitness


def evaluate_population(genomes, envs, config, fitness_function, batch_size=10):
    # evaluates population on set of environments
    count = 0
    env_fitness = np.zeros((len(genomes), len(envs)))
    fitness = np.zeros(len(genomes))

    for i, env in enumerate(envs):
        count = 0
        while (count < len(genomes)):
            batch = genomes[count:count + batch_size]
            batch_fitness = eval_batch_on_env(batch, env, config, fitness_function)
            env_fitness[count:count+batch_size, i] = batch_fitness
            count += batch_size

    for index, genome in enumerate(genomes):
        genome.env_fitness = env_fitness[index, :]

    fitness = np.mean(env_fitness, axis=1)
    return fitness


def evaluate_one(genome, envs, config, fitness_function, show=False, jc=None, cc=None):
    # evaluates one genome in one environment
    # gives possibility of showing
    tree_kwargs = configtools.tree_from_config(config)
    sim_kwargs = configtools.sim_from_config(config)
    envs_kwargs = configtools.envs_from_config(config)
    assert isinstance(envs, list), 'envs must be list'
    env_fitness = np.zeros(len(envs))

    for i, env in enumerate(envs):
        t = treenome.Tree(**tree_kwargs)

        sim = pyrosim.Simulator(play_blind=not show,
                                **sim_kwargs)
        # send tree body and network
        genome.send_to_simulator(sim, t)
        # send environment
        environments.send_to_simulator(sim, env, t, **envs_kwargs)

        sim.start()
        sensor_data = sim.wait_to_finish()
        env_fitness[i] = fitness_function(env, sensor_data, 20)

    # return np.average(env_fitness) - genome.calc_tree_joint_cost(), env_fitness
    return np.mean(env_fitness), env_fitness


def lookat_function(env, sensor_data, n_time_steps=1):
    """calculates how much robot is looking at objects correctly"""

    # sensor data of cylinders in the environment
    seen_sensor_data = sensor_data[-len(env):, 0, :]
    # modifier term specifying which cylinders are which
    env_multiplier = np.zeros(len(env))

    # 'look at' specifier for each cylinder
    for i, cyl in enumerate(env):
        if cyl == '0': #look away
            env_multiplier[i] = -1.0
        elif cyl == '1': # look towards
            env_multiplier[i] = 1.0

    # sum of the sensor values of each environment from 
    # the last n_time_steps. 0 for not unseen, 1 for seen
    total_seen = np.sum(seen_sensor_data[:, -n_time_steps:], axis=1)

    # sum of the sensors multiplied by their env modifier to get 
    # a raw fitness score.
    raw_fitness = np.dot(total_seen, env_multiplier)
    ones = env.count('1')
    zeros = env.count('0')

    # create normalizer based on total maximum and minimum score
    max_possible = ones * n_time_steps
    min_possible = zeros * -n_time_steps
    
    # normalize the fitness to [0, 1]
    normalized_fitness = (raw_fitness - min_possible) / (max_possible - min_possible)
    return normalized_fitness

def mutate_genome_weights(genome):
    # runs genome mutation function
    genome.mutate(weights=True, expression=False, joints=False)
    return genome


def mutate_genome_topology(genome):
    genome.mutate(weights=True, expression=True, joints=False)
    return genome


def mutate_genome_all(genome):
    genome.mutate(weights=True, expression=True, joints=True)
    return genome


if __name__ == '__main__':
    import sys
    import os

    mutation_funcs = [mutate_genome_weights, mutate_genome_topology,
                      mutate_genome_all]
    mutation_names = ['weights', 'topology', 'all']

    cost_types = ['none', 'cc', 'jc']

    config_file = 'base-tree.config'
    gens = 10
    pop_size = 10
    experiment = 0
    save_dir = 'data/junk/'
    mutation_index = 2
    cost_index = 0
    seed = 0
    input_args = sys.argv
    n = 1
    if len(input_args) > n:
        seed = int(input_args[n])
    n += 1
    if len(input_args) > n: # experiment
        mutation_index = int(input_args[n])
    n += 1
    if len(input_args) > n:
        cost_index = int(input_args[n])
    n += 1
    if len(input_args) > n: # pop size
        pop_size = int(input_args[n])
    n += 1
    if len(input_args) > n: # num gens
        gens = int(input_args[n])
    n += 1    
    if len(input_args) > n: # save directory
        save_dir = input_args[n]
    n += 1
    if len(input_args) > n: # config file
        config_file = input_args[n]
    n += 1
    # envs = ['0000', '1111']
    envs = ['0000', '0001', '0010', '0011', '0100', '0101',
            '0110', '0111', '1000', '1001', '1010', '1011',
            '1100', '1101', '1110', '1111']

    np.random.seed(seed)
    random.seed(seed)

    # create checkpoint directory
    os.makedirs(save_dir + 'checkpoint', exist_ok=True)

    cost = cost_types[cost_index]
    mutation_func = mutation_funcs[mutation_index]

    config = configtools.config_from_file(config_file)

    genome_gen_func = functools.partial(create_genome, config)

    eval_func = functools.partial(evaluate_population,
                                    envs=envs,
                                    config=config,
                                    fitness_function=lookat_function)

    evolver = evolver.Afpo(pop_size=pop_size,
                           genome_gen_func=genome_gen_func,
                           eval_func=eval_func,
                           mutation_func=mutation_func)

    save_name = mutation_names[mutation_index] + '-' +  \
                cost + '-' + str(seed)

    checkpoint = 5
    
    fitness = []

    for gen in range(gens):
        evolver.evolve_for_one_step()

        if gen % checkpoint == 0 and gen != 0:
            # checkpoint population
            checkpoint_file_name = save_dir + 'checkpoint/' + \
                                   'Checkpoint-' + \
                                   str(gen) + '-' + save_name + '.pickle'

            with open(checkpoint_file_name, 'wb') as f:
                pickle.dump(evolver.population, f)

            fitness.append(evolver.population[0].fitness)

    # save dictionary
    data = {}
    data['fitness'] = fitness
    data['best'] = evolver.population[0]
    data['config'] = config
    data['pop_size'] = pop_size
    data['gens'] = gens

    # save best
    save_name = save_dir + save_name + '.pickle'

    with open(save_name, 'wb') as f:
        pickle.dump(data, f)


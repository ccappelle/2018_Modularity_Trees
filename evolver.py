from __future__ import division, print_function
import copy
import time


class Individual(object):
    def __init__(self, genome, id_tag, lineage, fitness=None):
        self.genome = genome
        self.id_tag = id_tag
        self.fitness = fitness
        self.lineage = lineage

    def __str__(self):
        for attribute, value in self.__dict__.items():
            print(attribute, value)
        return ''

class Evolver(object):
    """Evolver base class

    Basic base class which maintains a populations genome,
    id, and fitness.

    Attributes
    ----------
    population      : dictionary
        The population of genomes. This base class provides
        three standard keys to the dictionary: 'genome', 'my_id', and
        'fitness'. New keys can be added for more complex evolutionary
        algorithms.

    genome_gen_func : function
        The genome generator function. When a new individual is born,
        It is created with this generator function.

    eval_func       : function
        Evaluation function. Should take in a list of genomes and output
        a fitness for each individual.

    mutation_func   : function
        Mutation function. Takes in a genome, mutates it, and returns it.
        Function must output the mutated genome.

    pop_size        : int
        The default size of the population.

    base_pop        : list of genomes
        A base starting population of genomes
    """

    def __init__(self, pop_size, genome_gen_func,
                 eval_func, mutation_func, save_func=None,
                 base_pop=None,
                 print_func=None):
        self._generation = 0
        self.genome_gen_func = genome_gen_func
        self.mutation_func = mutation_func
        self.pop_size = pop_size
        self.eval_func = eval_func

        self.fitness_per_gen = []
        self.population_per_gen = []

        if print_func is None:
            self.print_func = self.print_out
        else:
            self.print_func = print_func

        self._sorted = False
        self._next_id = 0
        self._next_lineage = 0

        self._start_time = time.clock()

        self.population = []

        # add from base population (aka checkpoint)
        if base_pop:
            for indv in base_pop:
                self.add_genome_to_pop(indv)
        else:
            for i in range(pop_size):
                self.add_random_indv()

    def add_random_indv(self, genome_gen_func=None):
        """Adds a new individual to the population with fitness=None

        Parameters
        ----------
        genome_gen_func : function
            The generator for the new genome. Default is to use the
            generator specified in the __init__() call.

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        assert genome_gen_func or self.genome_gen_func
        if not genome_gen_func:
            genome_gen_func = self.genome_gen_func
        genome = genome_gen_func()

        self.add_genome_to_pop(genome, self._next_lineage)
        self._next_lineage += 1
        return True

    def add_genome_to_pop(self, genome, lineage):
        """Adds a pre-specified genome to the population."""

        individual = Individual(genome, self._next_id, lineage)
        self._next_id += 1
        self.population.append(individual)
        return True

    def checkpoint(self, to_save):
        self._save_func(to_save)

    def copy_and_mutate(self, index, mutation_func=None):
        """Mutates the genome in the population at index"""
        assert mutation_func or self.mutation_func
        mutant = copy.deepcopy(self.population[index].genome)

        if not mutation_func:
            mutation_func = self.mutation_func

        mutant = mutation_func(mutant)

        return mutant

    def evaluate(self, eval_func=None):
        """Evaluates the un-evaluated genomes"""
        genomes_to_eval = []
        indexes = []

        if not eval_func:
            eval_func = self.eval_func

        for i, indv in enumerate(self.population):
            if (indv.fitness is None):
                genomes_to_eval.append(indv.genome)
                indexes.append(i)

        fitness_vals = eval_func(genomes_to_eval)

        for i, index in enumerate(indexes):
            fitness = fitness_vals[i]
            self.population[index].fitness = fitness

        return True

    def get_generation(self):
        return self._generation

    def set_generation(self, gen):
        self._generation = gen

    def sort_population(self, key=lambda k: k.fitness):
        self.population.sort(key=key)
        self._sorted = True

        return True

    def print_out(self):
        print(self._generation, self.population[0].fitness)

class Afpo(Evolver):
    def __init__(self, pop_size, genome_gen_func,
                 eval_func, mutation_func,
                 base_pop=None,
                 print_func=None):
        super(Afpo, self).__init__(pop_size, genome_gen_func,
                                   eval_func, mutation_func,
                                   base_pop, print_func)

    def add_random_indv(self, genome_gen_func=None):
        super(Afpo, self).add_random_indv(genome_gen_func)
        self.population[-1].age = 0
        self.population[-1].num_dominated_by = 0

    def add_genome_to_pop(self, genome, lineage, age=0):
        super(Afpo, self).add_genome_to_pop(genome, lineage)
        self.population[-1].age = age
        self.population[-1].num_dominated_by = 0

    def age_population(self):
        for indv in self.population:
            indv.age += 1

    def determine_dominated_by(self):
        """Determines how many other individuals dominate a genome"""

        # clear the num dominated by
        for indv in self.population:
            indv.num_dominated_by = 0

        # indv is dominated if older and poorer fitness than another
        for i in range(len(self.population)):
            indv1 = self.population[i]
            for j in range(i + 1, len(self.population)):
                indv2 = self.population[j]

                if (indv1.fitness > indv2.fitness):
                    if (indv1.age <= indv2.age):
                        indv2.num_dominated_by += 1

                elif (indv2.fitness > indv1.fitness):
                    if (indv2.age <= indv1.age):
                        indv1.num_dominated_by += 1

                elif (indv1.fitness == indv2.fitness):
                    if (indv1.age < indv2.age):
                        indv2.num_dominated_by += 1
                    elif (indv1.age > indv2.age):
                        indv1.num_dominated_by += 1
                    else:  # ages are equal
                        if (indv1.id_tag > indv2.id_tag):
                            indv2.num_dominated_by += 1
                        else:
                            indv1.num_dominated_by += 1

    def evolve(self, num_gens, fitness_threshold=None,
               save_func=None, checkpoint=None):
        """Evolves the population to the specified criteria"""
        assert num_gens is not None or fitness_threshold is not None, (
            'Must define number of '
            'generations to run or fitness '
            'threshold to achieve')

        if num_gens:
            while(self._generation < num_gens):
                self.evolve_for_one_step()

                # store current population
                self.population_per_gen.append(copy.copy(self.population))
                # store best fitness
                self.fitness_per_gen.append(self.population[0].fitness)

        # best_indvs = []
        # fitness_vector = []
        # # fitness_scores = []
        # self.evaluate()

        # if num_gens:
        #     while(self._generation < num_gens):
        #         # print (self._generation)
        #         best_indv = self.evolve_for_one_step()
        #         fitness_vector.append(best_indv.fitness)
        #         best_indvs.append(best_indv)
        #         if (save_func and self._generation > 0 and
        #                 self._generation % checkpoint == 0):
        #             save_func(best_indvs)
        #             print('checkpointed at ' + str(self._generation))
        # else:
        #     best_fitness = float('-inf')
        #     while(best_fitness < fitness_threshold):
        #         best_indv = self.evolve_for_one_step()
        #         best_fitness = best_indv.fitness
        #         best_indvs.append(best_indv)

        # return best_indv.fitness, best_indv.genome

    def evolve_for_one_step(self):
        # increment generation
        self._generation += 1
        self._sorted = False

        # mutate pop
        self.mutate_population()

        # age population
        self.age_population()

        # inject new genome
        self.add_random_indv()

        # evaluate population
        self.evaluate()

        # count dominated
        self.determine_dominated_by()

        # kill back to population size
        self.remove_dominated()

        # print out
        self.print_out()

        # return best
        # return self.population[0]

    def get_pareto_front(self):
        """returns the Pareto front of the population"""
        if not(self._sorted):
            self.sort_population()

        pareto_front = []
        index = 0

        while(index < len(self.population) and
                self.population[index].num_dominated_by == 0):
            pareto_front.append(self.population[index])
            index += 1
        return pareto_front

    def mutate_population(self, mutation_func=None):
        """Creates mutant for each indv in population"""

        pop_length = len(self.population)
        for i in range(pop_length):
            mutant = self.copy_and_mutate(i, mutation_func)
            self.add_genome_to_pop(mutant, self.population[i].lineage, age=self.population[i].age)

    def print_out(self):
        """prints out the current generation and best fitness"""
        pareto = len(self.get_pareto_front())
        best_fitness = self.population[0].fitness
        time_passed = int(time.clock() - self._start_time)
        minutes, seconds = divmod(time_passed, 60)
        hours, minutes = divmod(minutes, 60)

        out_string = ('{}: {}\n  {}: {},\n  {}: {},\n  '
                      '{} {}:{:02d}:{:02d}, '
                      ).format(
            'gen', self._generation,
            'best fitness', best_fitness,
            'Pareto size', pareto,
            'time', hours, minutes, seconds)

        print(out_string)

    def remove_dominated(self):
        """removes the dominated until pop is reduced to original size"""

        if (self._sorted is False):
            self.sort_population()

        # removes indvs from the back of the population
        for i in range(len(self.population) - 1, self.pop_size - 1, -1):
            del self.population[i]

    def sort_population(self):
        """sorts by dominated by and fitness"""
        super(Afpo, self).sort_population(key=lambda k:
                                          (k.num_dominated_by,
                                           -k.fitness))
        return True


if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    def gen_bit_array(n=100):
        array = []
        for i in range(n):
            if random.random() < 0.5:
                array.append(0)
            else:
                array.append(1)

        return array


    def mutate_array_func(array, p=.1):
        mutant = [0] * len(array)

        for i in range(len(array)):
            if random.random() < p:
                mutant[i] = abs(1 - array[i])
            else:
                mutant[i] = array[i]

        return mutant


    def eval_pop_of_arrays(arrays):
        fitness = []

        for array in arrays:
            fitness.append(eval_array(array))

        return fitness


    def eval_array(array):
        return sum(array)

    evolver = Afpo(10, gen_bit_array, eval_pop_of_arrays, mutate_array_func)
    evolver.evolve(num_gens=1000)

    pop_per_gen = evolver.population_per_gen
    fitness_by_id = {}

    # for gen, population in enumerate(pop_per_gen):
    #     for indv in population:
    #         id_tag = indv.lineage
    #         if id_tag in fitness_by_id:
    #             if fitness_by_id[id_tag]['end'] == gen:
    #                 best = max(fitness_by_id[id_tag]['fitness'][-1], indv.fitness)
    #                 fitness_by_id[id_tag]['fitness'][-1] = best
    #             else:
    #                 fitness_by_id[id_tag]['fitness'].append(indv.fitness)
    #                 fitness_by_id[id_tag]['end'] = gen
    #         else:
    #             fitness_by_id[id_tag] = {}
    #             fitness_by_id[id_tag]['fitness'] = [indv.fitness]
    #             fitness_by_id[id_tag]['start'] = gen
    #             fitness_by_id[id_tag]['end'] = gen


    # for id_tag in fitness_by_id:
        
    #     start = fitness_by_id[id_tag]['start']
    #     end = fitness_by_id[id_tag]['end']
    #     if start != end:
    #         y = fitness_by_id[id_tag]['fitness']
    #         x = np.linspace( start, end, num=len(y))
    #         plt.plot(x, y)

    # plt.show()

    # pareto front
    last_pop = evolver.population_per_gen[-1]

    fitnesses = []
    ages = []

    for indv in last_pop:
        fitnesses.append(indv.fitness)
        ages.append(indv.age)

    plt.scatter(ages, fitnesses)
    plt.show()
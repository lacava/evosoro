import numpy as np
import random
import math
# import pdb
# WGL: epsilon lexicase selection 

def mad(x, axis=None):
    """median absolute deviation statistic"""
    return np.median(np.abs(x - np.median(x, axis)), axis)

def epsilon_lexicase_selection(population):
    """Return a list of selected individuals from the population.

    Individuals are selected by filtering on randomized orderings of time steps for each parent selection. 
    Each selection:
    1. start with whole population
    2. choose a random time window
    3. filter out any individuals that aren't within epsilon of the best performance in that window
    4. while there are still individuals and cases, repeat 3 with another time window
    5. return a random ind from the pool
    Parameters
    ----------
    population : Population
        This provides the individuals for selection.

    Returns
    -------
    new_population : list
        A list of selected individuals.

    """
    new_pop = []
    # pdb.set_trace()
    #print ( 'n.deltaNormDist:', n.deltaNormDist)
    F = np.array( [n.deltaNormDist for n in population.individuals])
    print('F:', F)
    individual_locs = np.arange(len(population))
    # calculate epsilon thresholds based on median absolute deviation (MAD)
    mad_for_case = np.array([mad(f) for f in F.transpose()])
    for i in np.arange(population.pop_size):

        can_locs = individual_locs
        cases = list(np.arange(F.shape[1]))
        np.random.shuffle(cases)
        # pdb.set_trace()
        while len(cases) > 0 and len(can_locs) > 1:
            # get best fitness for case among candidates
            best_val_for_case = np.max(F[can_locs,cases[0]])
            # filter individuals without an elite fitness on this case
            can_locs = [l for l in can_locs if F[l,cases[0]] >= best_val_for_case - mad_for_case[cases[0]]]
            cases.pop(0)

        choice = np.random.randint(len(can_locs))
        new_pop.append(population[can_locs[choice]]) 
    
    for ind in population:
        if ind in new_pop:
            ind.selected = 1
        else:
            ind.selected = 0

    return new_pop #return [ind for ind in population if ind.selected==1]


def pareto_selection(population):
    """Return a list of selected individuals from the population.

    All individuals in the population are ranked by their level, i.e. the number of solutions they are dominated by.
    Individuals are added to a list based on their ranking, best to worst, until the list size reaches the target
    population size (population.pop_size).

    Parameters
    ----------
    population : Population
        This provides the individuals for selection.

    Returns
    -------
    new_population : list
        A list of selected individuals.

    """
    new_population = []

    # SAM: moved this into calc_dominance()
    # population.sort(key="id", reverse=False) # <- if tied on all objectives, give preference to newer individual

    # (re)compute dominance for each individual
    population.calc_dominance()

    # sort the population multiple times by objective importance
    population.sort_by_objectives()

    # divide individuals into "pareto levels":
    # pareto level 0: individuals that are not dominated,
    # pareto level 1: individuals dominated one other individual, etc.
    done = False
    pareto_level = 0
    while not done:
        this_level = []
        size_left = population.pop_size - len(new_population)
        for ind in population:
            if len(ind.dominated_by) == pareto_level:
                this_level += [ind]

        # add best individuals to the new population.
        # add the best pareto levels first until it is not possible to fit them in the new_population
        if len(this_level) > 0:
            if size_left >= len(this_level):  # if whole pareto level can fit, add it
                new_population += this_level

            else:  # otherwise, select by sorted ranking within the level
                new_population += [this_level[0]]
                while len(new_population) < population.pop_size:
                    random_num = random.random()
                    log_level_length = math.log(len(this_level))
                    for i in range(1, len(this_level)):
                        if math.log(i) / log_level_length <= random_num < math.log(i + 1) / log_level_length and \
                                        this_level[i] not in new_population:
                            new_population += [this_level[i]]
                            continue

        pareto_level += 1
        if len(new_population) == population.pop_size:
            done = True

    for ind in population:
        if ind in new_population:
            ind.selected = 1
        else:
            ind.selected = 0

    return new_population


def pareto_tournament_selection(population):
    """Reduce the population pairwise.

    Two individuals from the population are randomly sampled and the inferior individual is removed from the population.
    This process repeats until the population size is reduced to either the target population size (population.pop_size)
    or the number of non-dominated individuals / Pareto front (population.non_dominated_size).

    Parameters
    ----------
    population : Population
        This provides the individuals for selection.

    Returns
    -------
    new_population : list
        A list of selected individuals.

    """
    # population.add_random_individual()  # adding in random ind in algorithms.py
    population.calc_dominance()
    random.shuffle(population.individuals)
    print "The nondominated size is", population.non_dominated_size

    while len(population) > population.pop_size and len(population) > population.non_dominated_size:

        inds = random.sample(range(len(population)), 2)
        ind0 = population[inds[0]]
        ind1 = population[inds[1]]

        if population.dominated_in_multiple_objectives(ind0, ind1):
            print "(fit) {0} dominated by {1}".format(ind0.fitness, ind1.fitness)
            print "(age) {0} dominated by {1}".format(ind0.age, ind1.age)
            population.pop(inds[0])
        elif population.dominated_in_multiple_objectives(ind1, ind0):
            print "(fit) {1} dominated by {0}".format(ind0.fitness, ind1.fitness)
            print "(age) {1} dominated by {0}".format(ind0.age, ind1.age)
            population.pop(inds[1])
        # else:
        #     population.pop(random.choice(inds))

    population.sort_by_objectives()

    return population.individuals

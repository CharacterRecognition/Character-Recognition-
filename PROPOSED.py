import numpy as np
import time

def PROPOSED(population, obj_func, lb, ub, T):
    # Improved Carpet Weaver Optimization
    # current update is done at line 26
    N, dim = population.shape

    # Evaluate initial fitness
    fitness = np.array([obj_func(population[i, :]) for i in range(N)])

    best_fitness = np.inf
    best_solution = np.zeros(dim)
    convergence = np.zeros(T)

    start_time = time.time()

    for t in range(1, T + 1):

        for i in range(N):

            #Phase 1: Exploration
            # r = np.random.rand(dim)
            currentFit = fitness[i]
            worstFit = np.min(fitness)
            r = currentFit/(currentFit + worstFit)
            # x_p1 = lb(i,:) + r * (ub(i,:) - lb(i,:))
            x_p1 = lb[i, :] + r * (ub[i, :] - lb[i, :])
            x_p1 = np.clip(x_p1, lb[i, :], ub[i, :])

            # new_x_p1 = population(i,:) + (1-2*r)*(x_p1 - population(i,:))
            new_x_p1 = population[i, :] + (1 - 2 * r) * (x_p1 - population[i, :])
            new_fitness_p1 = obj_func(new_x_p1)

            if new_fitness_p1 < fitness[i]:
                population[i, :] = new_x_p1
                fitness[i] = new_fitness_p1

            #  Phase 2: Exploitation
            if t != 0:
                new_x_p2 = population[i, :] + (1 + ((1 - 2 * r) / t)) * population[i, :]
            else:
                new_x_p2 = population[i, :].copy()

            new_fitness_p2 = obj_func(new_x_p2)

            if new_fitness_p2 < fitness[i]:
                population[i, :] = new_x_p2
                fitness[i] = new_fitness_p2

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_solution = population[best_idx, :].copy()
        convergence[t - 1] = best_fitness

    elapsed_time = time.time() - start_time
    return best_fitness, convergence, best_solution, elapsed_time

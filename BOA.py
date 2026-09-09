import numpy as np
from time import time
from numpy.random import uniform


def simplebounds(s, lb, ub):
    # Apply the lower bound
    for i in range(len(s)):
        if s[i] < lb[i]:
            s[i] = lb[i]

        # Apply the upper bounds
        if s[i] > ub[i]:
            s[i] = ub[i]
    return s


def BOA(Positions, fobj, LB, UB, Max_iter):
    # Botox Optimization Algorithm (BOA)
    epoch = Max_iter
    pop_size = Positions.shape[0]
    c = 0.01  # 0.01, is the sensory modality
    p = 0.8  # 0.8, Search for food and mating partner by  can occur at both local and global scale
    alpha = 0.1  # 0.1-0.3 (0 -> vo cung), the power exponent dependent on modality
    lb = LB[0, :]
    ub = UB[0, :]
    Fitness = np.zeros((pop_size))
    for i in range(pop_size):
        Fitness[i] = fobj(Positions[i, :])

    # Find the current best_pos
    fmin = min(Fitness)
    I = np.where(min(Fitness) == Fitness)
    g_best = Positions[I[0], :]
    S = Positions

    pop = Positions
    c_temp = c

    Convergence_curve = np.zeros((epoch))
    best_positions = np.zeros((epoch, Positions.shape[1]))
    ct = time()
    for n in range(epoch):

        for i in range(pop_size):
            # Calculate fragrance of which is correlated with objective function
            Fnew = fobj(S[i, :])
            FP = c_temp * (Fnew[0][0] ** alpha)
            t1 = None
            if uniform() < p:
                t1 = pop[i, :] + (uniform() * uniform() * g_best - pop[i, :]) * FP
            else:
                epsilon = uniform()
                id1, id2 = np.random.randint(0, pop_size, 2)
                dis = (epsilon ** 2) * pop[id1, :] - pop[id2, :]
                t1 = pop[i, :] + dis * FP
            # Check if the simple limits / bounds are OK
            t1 = simplebounds(t1.ravel(), lb, ub)
            fit = fobj(t1)  # Fnew represents new fitness values

            if fit < Fitness[i]:
                Fitness[i] = fit
                pop[i, :] = t1

            if fit < fmin:
                g_best = S[i, :]
                fmin = fit
        Convergence_curve[n] = fmin
        best_positions[n, :] = g_best
        c_temp = c_temp + 0.025 / (c_temp * epoch)
    ct = time() - ct
    return fmin.ravel(), Convergence_curve, best_positions[-1, :], ct

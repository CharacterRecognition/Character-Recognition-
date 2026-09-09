import numpy as np
import time

def apply_bounds(x, xmin, xmax):
    return np.clip(x, xmin, xmax)


def ECO(X, fname, xmin, xmax, Max_iter):
    # EDUCATIONAL COMPETITION OPTIMIZER
    N, dim = X.shape
    # fitness evaluation
    fitness = np.array([fname(X[i, :]) for i in range(N)])
    best_idx = np.argmin(fitness)
    best_sol = np.copy(X[best_idx])
    best_fit = fitness[best_idx]

    start_time = time.time()
    for it in range(Max_iter):

        # Find best & worst
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)

        best_sol = np.copy(X[best_idx])
        worst_sol = np.copy(X[worst_idx])

        R1 = np.random.rand()
        R2 = np.random.rand()
        P  = np.random.rand()
        E  = np.random.rand()

        #Student competition loop

        for j in range(N):

            # Determine stage (Primary / Middle / High school)
            stage = j % 3

            # Stage 1: PRIMARY SCHOOL

            if stage == 1:
                if j < N/2:
                    # Eq (3)
                    X[j] = X[j] + R1 * (best_sol - abs(X[j]))
                else:
                    # Eq (4)
                    X[j] = X[j] + R2 * (abs(worst_sol) - X[j])


            # Stage 2: MIDDLE SCHOOL
            elif stage == 2:
                if j < N/2:
                    # Eq (10)
                    X[j] = X[j] + P * (best_sol - X[j])
                else:
                    # Eq (11)
                    X[j] = X[j] - P * (X[j] - worst_sol)

            #Stage 3: HIGH SCHOOL
            else:
                if j < N/2:
                    # Eq (12)
                    X[j] = X[j] + E * (best_sol - 2*X[j])
                else:
                    # Eq (13)
                    X[j] = X[j] + E * (worst_sol - X[j])

            # apply bounds
            X[j] = apply_bounds(X[j], xmin, xmax)

        new_fitness = np.array([fname(X[i]) for i in range(N)])

        for i in range(N):
            if new_fitness[i] > fitness[i]:   # choose positive greedy
                new_fitness[i] = fitness[i]
                X[i] = X[i]

        fitness = new_fitness

    best_idx = np.argmin(fitness)
    bestfit3 = fitness[best_idx]
    bestsol3 = X[best_idx]
    fitness3 = fitness
    time3 = time.time() - start_time
    return bestfit3, fitness3, bestsol3, time3

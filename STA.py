import numpy as np
import time


def STA(X, func, lb, ub, max_iter):
    # Supercell Thunderstorm Algorithm (STA)
    TF = 0.7
    minimize = True
    seed = None
    rng = np.random.default_rng(seed)
    pop_size, dim = X.shape

    # Evaluate fitness
    F = np.array([func(x) for x in X])
    best_idx = np.argmin(F) if minimize else np.argmax(F)
    Xbest = X[best_idx].copy()
    Fbest = F[best_idx]

    convergence = np.zeros(max_iter)
    ct = time.time()

    for t in range(1, max_iter + 1):
        # Compute mean group (MG)
        MGi = np.mean(X, axis=0)

        for i in range(pop_size):
            Xi = X[i].copy()

            # Random control parameters
            a = rng.random()
            c = rng.random()

            # Weight coefficients (Eq.2)
            a1 = a + (1 - a) * (t / max_iter)
            a2 = (1 - a) - (1 - a) * (t / max_iter)
            ecs = np.e ** 3
            s = ecs + (1 - a) * (t / max_iter)
            b1 = ecs * np.cos(2 * np.pi * c)
            b2 = ecs * np.sin(2 * np.pi * c)

            # Spiral Motion stage (Eq.2)
            if rng.random() < (t / max_iter):
                Xnew = a1 * Xbest + b1 * np.abs(Xbest - Xi) + a2 * Xi
            else:
                Xnew = a1 * Xbest + b2 * np.abs(Xbest - MGi) + a2 * Xi

            # Tornado Formation stage (Eq.3)
            G1 = rng.random()
            rand_exp = rng.random()
            r2 = rng.random()
            idx = rng.choice(pop_size, 4, replace=False)
            r3, r4, r5, r6 = idx

            X_tornado = Xi + rng.random() * np.exp(-G1 * (t / max_iter)) * (X[r3] - X[r4]) \
                        + TF * (1 - rng.random()) * (X[r5] - X[r6])

            # Jet Stream stage (Eq.4)
            r7 = rng.random()
            if r7 <= 0.3:
                dr = rng.random(dim) - np.sinh(rng.random(dim))
                yr = rng.random(dim) - np.cosh(rng.random(dim))
                d1 = dr / (np.max(np.abs(dr)) + 1e-12)
                y1 = yr / (np.max(np.abs(yr)) + 1e-12)
                Xjet = rng.random() * Xbest + d1 * (Xi - 3 * rng.random() * MGi) + y1 * (Xi - 2 * Xbest)
            else:
                k = rng.integers(pop_size)
                Xjet = Xi + rng.random() * np.abs(Xbest - X[k]) + rng.random() * np.abs(Xi - X[k])

            # Combine all three phases
            Xcand = (Xnew + X_tornado + Xjet) / 3.0
            Xcand = np.clip(Xcand, lb[i], ub[i])

            Fnew = func(Xcand)
            if (minimize and Fnew < F[i]) or (not minimize and Fnew > F[i]):
                X[i] = Xcand
                F[i] = Fnew

        # Update best
        best_idx = np.argmin(F) if minimize else np.argmax(F)
        if (minimize and F[best_idx] < Fbest) or (not minimize and F[best_idx] > Fbest):
            Fbest = F[best_idx]
            Xbest = X[best_idx].copy()

        convergence[t - 1] = Fbest
    ct = time.time() - ct
    return Fbest, convergence, Xbest, ct


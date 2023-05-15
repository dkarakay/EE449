import os
import pickle

import numpy as np
from hw2 import Population, Individual, Gene


def ensure_non_decreasing(values):
    """Ensure that a list of values is non-decreasing."""
    for i in range(1, len(values)):
        values[i] = max(values[i], values[i - 1])
    return values


subfolder = "results"

# Get all folders names
folders = [
    f for f in os.listdir(subfolder) if os.path.isdir(os.path.join(subfolder, f))
]

folders.sort()
for folder in folders:
    if folder.startswith("default"):
        all = []
        p = pickle.load(
            open(f"results/{folder}/{folder}_10000.pkl", "rb")
        )  # type: Population

        fitnesses = [i.fitness for i in p.population]
        x = np.sort(fitnesses)
        y = sorted(p.population, key=lambda x: x.fitness, reverse=True)
        print(folder, ":", len(p.best_population))

        for i in range(len(p.best_population)):
            all.append(p.best_population[i].fitness)

        all = ensure_non_decreasing(all)

        # Print values each 100 generations
        for i in range(0, len(all), 100):
            print(i, all[i])

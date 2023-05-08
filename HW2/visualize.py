import pickle
import matplotlib.pyplot as plt
import numpy as np
from hw2 import Population, Individual, Gene


all = []


values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]
# values = [1000]
for val in values:
    p = pickle.load(
        open(f"results/default/default_{val}.pkl", "rb")
    )  # type: Population

    fitnesses = [i.fitness for i in p.population]
    x = np.sort(fitnesses)
    y = sorted(p.population, key=lambda x: x.fitness, reverse=True)
    print(f"Best fitness: {x[-1]}")
    print(f"Best individual: {y[0].fitness}")
    all.append(y[0].fitness)

print(all.sort())


plt.plot(values, all)
plt.legend(values)
plt.show()

import pickle

from hw2 import Population, Individual, Gene

all = []


values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
for val in values:
    p = pickle.load(
        open(f"results/default_backup/default_{val}.pkl", "rb")
    )  # type: Population

    fitnesses = [i.fitness for i in p.population]
    for f in fitnesses:
        all.append(f)

print(all.sort())

import matplotlib.pyplot as plt

plt.plot(all, values)
plt.legend(values)
plt.show()

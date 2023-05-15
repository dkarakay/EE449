# Created by Deniz Karakay on 30.04.2023
# Description: This file contains the main program for the second homework of EE449 course.
import copy
import datetime
import os
import random
import shutil
import time
import cv2
import numpy as np
import pickle

IMG_PATH = "painting.png"
IMG_WIDTH = 0
IMG_HEIGHT = 0


class Gene:
    def __init__(self, id=-2):
        self.id = id
        self.radius = random.randint(1, max(IMG_WIDTH, IMG_HEIGHT) // 2)
        self.x, self.y = self.determine_center_coordinates()
        self.red = random.randint(0, 255)
        self.green = random.randint(0, 255)
        self.blue = random.randint(0, 255)
        self.alpha = random.random()

    def is_valid_circle(self, x, y):
        # Check if the circle is completely outside the image
        if x - self.radius >= IMG_WIDTH or x + self.radius < 0:
            return False
        if y - self.radius >= IMG_HEIGHT or y + self.radius < 0:
            return False

        # Check if the circle intersects with the image
        if x - self.radius < 0 or x + self.radius >= IMG_WIDTH:
            return True
        if y - self.radius < 0 or y + self.radius >= IMG_HEIGHT:
            return True

        # Check if the circle is completely inside the image
        if x - self.radius >= 0 and x + self.radius < IMG_WIDTH:
            return True
        if y - self.radius >= 0 and y + self.radius < IMG_HEIGHT:
            return True

        return False

    # Determine the center coordinates of the gene
    def determine_center_coordinates(self, guided=False):
        while True:
            if guided:
                x = self.x + random.randint(-IMG_WIDTH // 4, IMG_WIDTH // 4)
                y = self.y + random.randint(-IMG_HEIGHT // 4, IMG_HEIGHT // 4)
            else:
                x = random.randint(IMG_WIDTH * -1.6, IMG_WIDTH * 1.6)
                y = random.randint(IMG_HEIGHT * -1.6, IMG_HEIGHT * 1.6)

            if self.is_valid_circle(x, y):
                return x, y

    # Mutate the gene with a guided mutation
    def guided_mutation(self):
        self.radius = np.clip(
            self.radius + random.randint(-10, 10), 1, min(IMG_WIDTH, IMG_HEIGHT) // 2
        )

        # Determine the center coordinates of the gene
        self.x, self.y = self.determine_center_coordinates(guided=True)

        # Mutate the color and alpha of the gene
        self.red = int(np.clip(self.red + random.randint(-64, 64), 0, 255))
        self.green = int(np.clip(self.green + random.randint(-64, 64), 0, 255))
        self.blue = int(np.clip(self.blue + random.randint(-64, 64), 0, 255))
        self.alpha = np.clip(self.alpha + random.uniform(-0.25, 0.25), 0, 1)

    # Mutate the gene with a guided mutation
    def more_guided_mutation(self):
        self.radius = np.clip(
            self.radius + random.randint(-20, 20), 1, min(IMG_WIDTH, IMG_HEIGHT) // 2
        )

        # Determine the center coordinates of the gene
        self.x, self.y = self.determine_center_coordinates(guided=True)

        # Mutate the color and alpha of the gene
        self.red = int(np.clip(self.red + random.randint(-128, 128), 0, 255))
        self.green = int(np.clip(self.green + random.randint(-128, 128), 0, 255))
        self.blue = int(np.clip(self.blue + random.randint(-128, 128), 0, 255))
        self.alpha = np.clip(self.alpha + random.uniform(-0.5, 0.5), 0, 1)

    # Mutate the gene with a guided mutation
    def less_guided_mutation(self):
        self.radius = np.clip(
            self.radius + random.randint(-5, 5), 1, min(IMG_WIDTH, IMG_HEIGHT) // 2
        )

        # Determine the center coordinates of the gene
        self.x, self.y = self.determine_center_coordinates(guided=True)

        # Mutate the color and alpha of the gene
        self.red = int(np.clip(self.red + random.randint(-32, 32), 0, 255))
        self.green = int(np.clip(self.green + random.randint(-32, 32), 0, 255))
        self.blue = int(np.clip(self.blue + random.randint(-32, 32), 0, 255))
        self.alpha = np.clip(self.alpha + random.uniform(-0.12, 0.12), 0, 1)

    # Print every property of the gene in one line
    def print(self):
        print(
            "Gene ID:",
            self.id,
            "R:",
            self.radius,
            "X:",
            self.x,
            "Y:",
            self.y,
            "RGBA:",
            self.red,
            self.green,
            self.blue,
            self.alpha.__round__(2),
        )


class Individual:
    def __init__(self, num_genes, id=-2, fitness=0, chromosome=[]):
        self.id = id
        self.chromosome = chromosome
        self.fitness = fitness

        if len(chromosome) == 0:
            # Create a chromosome with num_genes
            for i in range(num_genes):
                self.chromosome.append(Gene(i))

            # Sort chromosomes by radius by descending order
            self.chromosome.sort(key=lambda x: x.radius, reverse=True)

    # Draw the individual to the screen
    def draw(self):
        # Create a white image
        img = np.full((IMG_WIDTH, IMG_HEIGHT, 3), 255, np.uint8)

        for gene in self.chromosome:
            # Create an overlay image
            overlay = img.copy()

            color = (gene.red, gene.green, gene.blue)

            # Draw the gene to the image
            cv2.circle(overlay, (gene.x, gene.y), gene.radius, color, -1)

            img = cv2.addWeighted(overlay, gene.alpha, img, 1 - gene.alpha, 0, img)

        return img

    # Evaluate the fitness of the individual
    def eval(self):
        # Convert source and individual to numpy arrays
        source = cv2.imread(IMG_PATH)
        source_np = np.array(source, dtype=np.int64)

        individual = self.draw()
        individual_np = np.array(individual, dtype=np.int64)

        # Calculate the difference between source and individual
        diff = np.subtract(source_np, individual_np)

        # Square the difference
        squared = np.square(diff)

        # Sum the squared difference
        self.fitness = -np.sum(squared)

        return self.fitness

    # Mutate the individual with mutation_type
    def mutate(self, mutation_type, mutation_prob):
        mutations = []
        while random.random() < mutation_prob:
            random_gene_id = random.randint(0, len(self.chromosome) - 1)

            if len(mutations) >= len(self.chromosome):
                return

            while random_gene_id in mutations:
                random_gene_id = random.randint(0, len(self.chromosome) - 1)

            mutations.append(random_gene_id)
            if mutation_type == "unguided":
                self.chromosome[random.randint(0, len(self.chromosome) - 1)] = Gene(
                    id=random_gene_id
                )
            elif mutation_type == "guided":
                self.chromosome[random_gene_id].guided_mutation()
            elif mutation_type == "more_guided":
                self.chromosome[random_gene_id].more_guided_mutation()
            elif mutation_type == "less_guided":
                self.chromosome[random_gene_id].less_guided_mutation()

    # Print every property of the individual in one line
    def print(self):
        print(
            "Individual ID:",
            self.id,
            "Fitness:",
            self.fitness,
            "Genes:",
            len(self.chromosome),
        )
        for gene in self.chromosome:
            gene.print()


class Population:
    def __init__(self, num_inds, num_genes, population=[]):
        self.population = population
        self.best_population = []

        if len(population) == 0:
            # Create a population with num_inds individuals
            for i in range(num_inds):
                self.population.append(Individual(id=i, num_genes=num_genes))

    # Evaluate every individual in the population
    def evaluate(self):
        for individual in self.population:
            individual.eval()

    # Sort the population by fitness in descending order
    def sort_population(self):
        t = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return t

    # Sort the population by fitness in descending order with respect to the given population
    def sort_inds(self, pop):
        return sorted(pop, key=lambda x: x.fitness, reverse=True)

    # Select the best num_elites individuals in the population
    # Return the best num_elites individuals and the rest of the population
    def selection(self, fraction_elites, fraction_parents, tm_size):
        self.population = self.sort_population()

        # Select the best num_elites individuals in the population
        num_elites = int(len(self.population) * fraction_elites)

        # Select the best num_parents individuals in the population
        num_parents = int(len(self.population) * fraction_parents)

        # If the number of parents is odd, make it even
        if num_parents % 2 == 1:
            num_parents += 1

        # Divide the population into elites and non_elites
        elites = self.population[:num_elites]
        non_elites = self.population[num_elites:]

        # Select num_parents individuals from the non_elites using tournament selection
        parents = []
        for i in range(num_parents):
            tournament = random.sample(non_elites, min(tm_size, len(non_elites)))
            best = self.sort_inds(tournament)[0]
            parents.append(best)
            non_elites.remove(best)

        return elites, non_elites, parents

    # Select two parents from the population using tournament selection
    def crossover(self, parents, num_genes):
        children = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            chromosome_child1 = []
            chromosome_child2 = []

            for j in range(len(parent1.chromosome)):
                # 50% chance to get the gene from parent1 or parent2
                if random.random() >= 0.5:
                    chromosome_child1.append(copy.deepcopy(parent1.chromosome[j]))
                    chromosome_child2.append(copy.deepcopy(parent2.chromosome[j]))
                # 50% chance to get the gene from parent2 or parent1
                else:
                    chromosome_child1.append(copy.deepcopy(parent2.chromosome[j]))
                    chromosome_child2.append(copy.deepcopy(parent1.chromosome[j]))

            #   Create two children with the same length as parents
            child1 = Individual(num_genes=num_genes, chromosome=chromosome_child1)
            child2 = Individual(num_genes=num_genes, chromosome=chromosome_child2)
            child1.eval()
            child2.eval()

            pop = Population.sort_inds(self, [child1, child2, parent1, parent2])
            children.append(pop[0])
            children.append(pop[1])

        return children

    # Print every property of the population in one line
    def print(self):
        for individual in self.population:
            individual.print()


# Save the population to a file using pickle
def save_population(population, path):
    with open(path + ".pkl", "wb") as f:
        pickle.dump(population, f, pickle.HIGHEST_PROTOCOL)


# Print the elapsed time in a readable format
def print_elapsed_time(elapsed_time):
    seconds = elapsed_time.seconds
    minutes = seconds // 60 % 60
    hours = seconds // 3600 % 3600

    if elapsed_time < datetime.timedelta(minutes=1):
        return f"{seconds} secs"
    elif elapsed_time < datetime.timedelta(hours=1):
        return f"{minutes} mins {seconds % 60} secs"
    else:
        return f"{hours} hours {minutes % 60} mins {seconds % 60} secs"


# Save the population and the best individual
def save_all(pop, name, generation, path, best_population, image_only=True):
    print(best_population.fitness)

    current_name = f"{name}_{generation}"
    file_path = f"{path}{current_name}"
    if not image_only:
        save_population(pop, file_path)

    cv2.imwrite(f"{path}{current_name}.png", best_population.draw())


# Run the evolutionary algorithm for the given parameters
def evolutionary_algorithm(
    name,
    num_generations=10000,
    num_inds=20,
    num_genes=50,
    tm_size=5,
    mutation_type="guided",
    fraction_elites=0.2,
    fraction_parents=0.6,
    mutation_prob=0.2,
):
    path = f"results/{name}/"

    if os.path.exists(path):
        shutil.rmtree(path)

    if not os.path.exists(path):
        os.mkdir(path)

    pop = Population(num_inds, num_genes)

    for generation in range(num_generations):
        pop.evaluate()

        # Select the best num_elites individuals in the population and the rest of the population
        elites, non_elites, parents = pop.selection(
            fraction_elites, fraction_parents, tm_size
        )

        # Create children from the parents
        children = pop.crossover(parents, num_genes)

        # Mutate the children and non_elites
        for individual in non_elites + children:
            individual.mutate(mutation_type, mutation_prob)

        # Add the elites, children and non_elites to the population
        pop.population = elites + children + non_elites

        sorted_population = pop.sort_population()
        pop.best_population.append(sorted_population[0])

        if generation % 1000 == 0 or generation == 0:
            save_all(pop, name, generation, path, sorted_population[0])

        if generation % 100 == 0:
            elapsed_time = datetime.datetime.now() - start_time
            print(
                "Generation:",
                generation,
                "Time:",
                print_elapsed_time(elapsed_time),
                "Fitness:",
                sorted_population[0].fitness,
            )

    pop.evaluate()
    sorted_population = pop.sort_population()
    pop.best_population.append(sorted_population[0])
    save_all(pop, name, 10000, path, sorted_population[0], image_only=False)


# Run the evolutionary algorithm for suggestions
def evolutionary_algorithm_for_suggestions(
    name,
    num_generations=10000,
    num_inds=20,
    num_genes=50,
    tm_size=5,
    mutation_type="guided",
    fraction_elites=0.2,
    fraction_parents=0.6,
    mutation_prob=0.2,
    suggestion_type="changing_mutation",
):
    path = f"results/{name}/"

    if os.path.exists(path):
        shutil.rmtree(path)

    if not os.path.exists(path):
        os.mkdir(path)

    pop = Population(num_inds, num_genes)

    for generation in range(num_generations):
        # Changing mutation probability
        if suggestion_type == "changing_mutation":
            if generation == 1:
                fraction_parents = 0.7
            elif generation == 750:
                mutation_prob = 0.6
            elif generation == 2000:
                mutation_prob = 0.5
            elif generation == 4000:
                mutation_prob = 0.4
            elif generation == 8000:
                mutation_prob = 0.3
            elif generation == 9000:
                mutation_prob = 0.2

        # Changing fraction of parents and elites
        elif suggestion_type == "changing_fraction_parents_elites":
            if generation == 1:
                fraction_parents = 0.75
                fraction_elites = 0.03
            elif generation == 750:
                fraction_parents = 0.7
                fraction_elites = 0.05
            elif generation == 2000:
                fraction_parents = 0.65
                fraction_elites = 0.08
            elif generation == 4000:
                fraction_parents = 0.6
                fraction_elites = 0.1
            elif generation == 8000:
                fraction_parents = 0.5
                fraction_elites = 0.2
            elif generation == 9000:
                fraction_parents = 0.4
                fraction_elites = 0.25

        # Changing mutation type
        elif suggestion_type == "changing_mutation_type":
            if generation == 1:
                mutation_type = "unguided"
            elif generation == 750:
                mutation_type = "less_guided"
            elif generation == 3000:
                mutation_type = "guided"
            elif generation == 6000:
                mutation_type = "more_guided"

        pop.evaluate()

        # Select the best num_elites individuals in the population and the rest of the population
        elites, non_elites, parents = pop.selection(
            fraction_elites, fraction_parents, tm_size
        )

        # Create children from the parents
        children = pop.crossover(parents, num_genes)

        # Mutate the children and non_elites
        for individual in non_elites + children:
            individual.mutate(mutation_type, mutation_prob)

        # Add the elites, children and non_elites to the population
        pop.population = elites + children + non_elites

        # Sort the population by fitness
        sorted_population = pop.sort_population()
        pop.best_population.append(sorted_population[0])

        # Save the best individual and the population in every 1000 generations
        if generation % 1000 == 0 or generation == 0:
            save_all(pop, name, generation, path, sorted_population[0])

        # Print the generation, elapsed time and fitness in every 100 generations
        if generation % 100 == 0:
            elapsed_time = datetime.datetime.now() - start_time
            print(
                "Generation:",
                generation,
                "Time:",
                print_elapsed_time(elapsed_time),
                "Fitness:",
                sorted_population[0].fitness,
            )

    # Evaluate the population one last time
    pop.evaluate()
    sorted_population = pop.sort_population()
    pop.best_population.append(sorted_population[0])
    save_all(pop, name, 10000, path, sorted_population[0], image_only=False)


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    img = cv2.imread(IMG_PATH)
    IMG_WIDTH = img.shape[0]
    IMG_HEIGHT = img.shape[1]

    IMG_RADIUS = (IMG_WIDTH + IMG_HEIGHT) / 2

    # evolutionary_algorithm_for_suggestions(
    #    name="tttt",
    #    mutation_prob=0.4,
    #    suggestion_type="changing_mutation_type",
    # )

    # evolutionary_algorithm_for_suggestions(
    #    name="suggestion_changing_mutation_type",
    #    mutation_prob=0.4,
    #    suggestion_type="changing_mutation_type",
    # )

    # evolutionary_algorithm_for_suggestions(
    #    name="suggestion_changing_mutation",
    #    suggestion_type="changing_mutation",
    # )

    # evolutionary_algorithm_for_suggestions(
    #    name="suggestion_changing_fraction_parents_elites",
    #    suggestion_type="changing_fraction_parents_elites",
    # )

    elapsed_time = datetime.datetime.now() - start_time
    print("All time:", print_elapsed_time(elapsed_time))

"""     evolutionary_algorithm(name="default")

    # # NUM_INDS = 5, 10, 40 and 60
    evolutionary_algorithm(name="num_inds_5", num_inds=5)
    evolutionary_algorithm(name="num_inds_10", num_inds=10)
    evolutionary_algorithm(name="num_inds_40", num_inds=40)
    evolutionary_algorithm(name="num_inds_60", num_inds=60)

    # # NUM_GENES = 15, 30, 80 and 120
    evolutionary_algorithm(name="num_genes_15", num_genes=15)
    evolutionary_algorithm(name="num_genes_30", num_genes=30)
    evolutionary_algorithm(name="num_genes_80", num_genes=80)
    evolutionary_algorithm(name="num_genes_120", num_genes=120)

    # TM_SIZE = 2, 8 and 16
    evolutionary_algorithm(name="tm_size_2", tm_size=2)
    evolutionary_algorithm(name="tm_size_8", tm_size=8)
    evolutionary_algorithm(name="tm_size_16", tm_size=16)

    # FRACTION_ELITES = 0.04 and 0.35
    evolutionary_algorithm(name="fraction_elites_0.04", fraction_elites=0.04)
    evolutionary_algorithm(name="fraction_elites_0.35", fraction_elites=0.35)

    # FRACTION_PARENTS = 0.15, 0.3 and 0.75
    evolutionary_algorithm(name="fraction_parents_0.15", fraction_parents=0.15)
    evolutionary_algorithm(name="fraction_parents_0.3", fraction_parents=0.3)
    evolutionary_algorithm(name="fraction_parents_0.75", fraction_parents=0.75)

    # MUTATION_PROB = 0.1, 0.4 and 0.75
    evolutionary_algorithm(name="mutation_prob_0.1", mutation_prob=0.1)
    evolutionary_algorithm(name="mutation_prob_0.4", mutation_prob=0.4)
    evolutionary_algorithm(name="mutation_prob_0.75", mutation_prob=0.75)

    # MUTATION_TYPE = "unguided"
    evolutionary_algorithm(name="mutation_type_unguided", mutation_type="unguided")

    elapsed_time = datetime.datetime.now() - start_time
    print("All time:", print_elapsed_time(elapsed_time)) """

# Created by Deniz Karakay on 30.04.2023
# Description: This file contains the main program for the second homework of EE449 course.
import copy
import os
import random
import shutil
import time
import cv2
import numpy as np

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
        # cv2.imshow("image", img)
        # cv2.waitKey(0)

        return img

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
                # print("Unguided Mutation")
                self.chromosome[random.randint(0, len(self.chromosome) - 1)] = Gene(
                    id=random_gene_id
                )
            elif mutation_type == "guided":
                # print("Guided Mutation")
                self.chromosome[random_gene_id].guided_mutation()

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

    # Sort the population by fitness in descending order
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
            tournament = random.sample(non_elites, tm_size)
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


def save_population(population, path):
    import pickle

    with open(path + ".pkl", "wb") as f:
        pickle.dump(population, f)


def evaluationary_algorithm(
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

        if generation % 1000 == 0 and generation != 0:
            sorted_population = pop.sort_population()
            print(sorted_population[0].fitness)
            pop.best_population.append(sorted_population[0])

            current_name = f"{name}_{generation}"
            file_path = f"{path}{current_name}"
            save_population(pop, file_path)

            cv2.imwrite(f"{path}{current_name}.png", sorted_population[0].draw())

        if generation % 100 == 0:
            print("Generation:", generation, "Time:", time.time() - start_time)

    pop.evaluate()
    sorted_population = pop.sort_population()
    pop.best_population.append(sorted_population[0])
    current_name = name + "_" + str(generation + 1)
    file_path = f"{path}{current_name}"
    save_population(pop, file_path)
    cv2.imwrite(f"{path}{current_name}.png", pop.best_population[0].draw())


if __name__ == "__main__":
    start_time = time.time()

    img = cv2.imread(IMG_PATH)
    IMG_WIDTH = img.shape[0]
    IMG_HEIGHT = img.shape[1]

    IMG_RADIUS = (IMG_WIDTH + IMG_HEIGHT) / 2
    # print(IMG_WIDTH, IMG_HEIGHT)

    evaluationary_algorithm(name="default")

    print("Execution time:", time.time() - start_time)

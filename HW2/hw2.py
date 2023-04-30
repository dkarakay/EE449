# Created by Deniz Karakay on 30.04.2023
# Description: This file contains the main program for the second homework of EE449 course.
import random
import cv2
import numpy as np

IMG_PATH = "painting.png"
IMG_WIDTH = 0
IMG_HEIGHT = 0
MUTATION_PROB = 0.2


class Gene:
    def __init__(self, id):
        self.id = id
        self.radius = random.randint(1, min(IMG_WIDTH, IMG_HEIGHT) // 2)
        self.x, self.y = self.check_valid_center()
        self.red = random.randint(0, 255)
        self.green = random.randint(0, 255)
        self.blue = random.randint(0, 255)
        self.alpha = random.random()

    def check_valid_center(self):
        while True:
            x = random.randint(-self.radius, IMG_WIDTH + self.radius)
            y = random.randint(-self.radius, IMG_HEIGHT + self.radius)

            if (x + self.radius) < IMG_WIDTH and (y + self.radius) < IMG_HEIGHT:
                return x, y

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
    def __init__(self, num_genes, id=-2, fitness=0):
        self.id = id
        self.chromosome = []
        self.fitness = fitness

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

            # Draw the gene to the image
            cv2.circle(
                overlay,
                (gene.x, gene.y),
                gene.radius,
                (gene.red, gene.green, gene.blue),
                -1,
            )
            img = cv2.addWeighted(overlay, gene.alpha, img, 1 - gene.alpha, 0, img)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)

        return img

    def eval(self):
        # Convert source and individual to numpy arrays
        source = cv2.imread(IMG_PATH)
        source_np = np.array(source, dtype=np.int32)

        individual = self.draw()
        individual_np = np.array(individual, dtype=np.int32)

        # Calculate the difference between source and individual
        diff = np.subtract(source_np, individual_np)

        # Square the difference
        squared = np.square(diff)

        # Sum the squared difference
        self.fitness = -np.sum(squared)

        return self.fitness

    def mutate(self, mutation_type):
        if mutation_type == "unguided":
            print("Unguided Mutation")

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
    def __init__(self, num_inds, num_genes):
        self.population = []

        # Create a population with num_inds individuals
        for i in range(num_inds):
            self.population.append(Individual(id=i, num_genes=num_genes))

    # Evaluate every individual in the population
    def evaluate(self):
        for individual in self.population:
            individual.eval()

    # Select the best num_elites individuals in the population
    # Return the best num_elites individuals and the rest of the population
    def selection(self, num_elites):
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return self.population[:num_elites], self.population[num_elites:]

    # Select the best individual in the population using tournament selection
    def tournament_selection(self, non_elites, tm_size):
        tournament = random.sample(non_elites, tm_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parents):
        parent1, parent2 = parents

        # Create two children with the same length as parents
        child1 = Individual(len(parent1.chromosome))
        child2 = Individual(len(parent2.chromosome))

        for i in range(len(parent1.chromosome)):
            # 50% chance to get the gene from parent1 or parent2
            if random.random() >= 0.5:
                child1.chromosome[i] = parent1.chromosome[i]
                child2.chromosome[i] = parent2.chromosome[i]

            # 50% chance to get the gene from parent2 or parent1
            else:
                child1.chromosome[i] = parent2.chromosome[i]
                child2.chromosome[i] = parent1.chromosome[i]
        return child1, child2

    # Print every property of the population in one line
    def print(self):
        for individual in self.population:
            individual.print()


def evaluationary_algorithm(
    num_inds, num_genes, num_elites=5, tm_size=5, num_parents=4
):
    population = Population(num_inds, num_genes)
    population.print()
    population.evaluate()

    # Select the best num_elites individuals in the population and the rest of the population
    elites, non_elites = population.selection(num_elites)

    parents = []
    for i in range(num_parents):
        parents.append(population.tournament_selection(non_elites, tm_size))

    print(len(parents))

    # Create children from the parents
    children = []
    for i in range(0, len(parents), 2):
        child1, child2 = population.crossover(parents[i : i + 2])
        children.append(child1)
        children.append(child2)

    print(len(children))

    best = max(population.population, key=lambda x: x.fitness)
    cv2.imshow("image", best.draw())
    cv2.waitKey(0)


if __name__ == "__main__":
    num_inds = 20
    num_genes = 50

    img = cv2.imread(IMG_PATH)
    IMG_WIDTH = img.shape[0]
    IMG_HEIGHT = img.shape[1]

    IMG_RADIUS = (IMG_WIDTH + IMG_HEIGHT) / 2
    print(IMG_WIDTH, IMG_HEIGHT)

    evaluationary_algorithm(num_inds=num_inds, num_genes=num_genes)

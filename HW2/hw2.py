# Created by Deniz Karakay on 30.04.2023
# Description: This file contains the main program for the second homework of EE449 course.
import random
import time
import cv2
import numpy as np

IMG_PATH = "painting.png"
IMG_WIDTH = 0
IMG_HEIGHT = 0
MUTATION_PROB = 0.2


class Gene:
    def __init__(self, id=-2):
        self.id = id
        self.radius = random.randint(1, min(IMG_WIDTH, IMG_HEIGHT) // 2)
        self.x, self.y = self.determine_center_coordinates()
        self.red = random.randint(0, 255)
        self.green = random.randint(0, 255)
        self.blue = random.randint(0, 255)
        self.alpha = random.random()

    def determine_center_coordinates(self, guided=False):
        while True:
            if guided:
                x = self.x + random.randint(-IMG_WIDTH // 4, IMG_WIDTH // 4)
                y = self.y + random.randint(-IMG_HEIGHT // 4, IMG_HEIGHT // 4)
            else:
                x = random.randint(-self.radius, IMG_WIDTH + self.radius)
                y = random.randint(-self.radius, IMG_HEIGHT + self.radius)

            if (x + self.radius) < IMG_WIDTH and (y + self.radius) < IMG_HEIGHT:
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

    def mutate(self, mutation_type):
        while random.random() < MUTATION_PROB:
            random_gene_id = random.randint(0, len(self.chromosome) - 1)

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
    def __init__(self, num_inds, num_genes):
        self.population = []
        self.best_population = []

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

    # Select the best num_elites individuals in the population
    # Return the best num_elites individuals and the rest of the population
    def selection(self, num_elites):
        self.population = self.sort_population()
        return self.population[:num_elites], self.population[num_elites:]

    # Select the best individual in the population using tournament selection
    def tournament_selection(self, non_elites, tm_size):
        tournament = random.sample(non_elites, tm_size)
        return max(tournament, key=lambda x: x.fitness)

    # Select two parents from the population using tournament selection
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
    num_inds,
    num_genes,
    num_generations,
    num_elites=5,
    tm_size=5,
    num_parents=4,
    mutation_type="guided",
):
    pop = Population(num_inds, num_genes)

    # population.print()
    for generation in range(num_generations):
        pop.evaluate()

        # Select the best num_elites individuals in the population and the rest of the population
        elites, non_elites = pop.selection(num_elites)

        parents = []
        for i in range(num_parents):
            parents.append(pop.tournament_selection(non_elites, tm_size))

        # Create children from the parents
        children = []
        for i in range(0, len(parents), 2):
            child1, child2 = pop.crossover(parents[i : i + 2])
            children.append(child1)
            children.append(child2)

        for individual in non_elites + children:
            individual.mutate(mutation_type)

        # Add the elites, children and non_elites to the population
        pop.population = elites + children + non_elites
        sorted_population = pop.sort_population()
        best = max(pop.population, key=lambda x: x.fitness)

        if generation % 100 == 0:
            sorted_population = pop.sort_population()
            pop.best_population.append(sorted_population[0])
            # for p in pop.best_population:
            #    name = self.name + "_" + str(p.id) + ".png"
            cv2.imwrite(f"results/{generation}.png", pop.best_population[0].draw())

    best = max(pop.population, key=lambda x: x.fitness)
    cv2.imshow("image", best.draw())
    cv2.waitKey(0)


if __name__ == "__main__":
    start_time = time.time()
    num_inds = 20
    num_genes = 50

    img = cv2.imread(IMG_PATH)
    IMG_WIDTH = img.shape[0]
    IMG_HEIGHT = img.shape[1]

    IMG_RADIUS = (IMG_WIDTH + IMG_HEIGHT) / 2
    print(IMG_WIDTH, IMG_HEIGHT)

    evaluationary_algorithm(
        num_inds=num_inds,
        num_genes=num_genes,
        num_generations=1000,
    )

    print("Execution time:", time.time() - start_time)

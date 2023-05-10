import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from hw2 import Population, Individual, Gene
from PIL import Image, ImageDraw, ImageFont


def combine_images():
    # Combine 11 images into a single image
    image_paths = [f"results/{folder}/{folder}_{i*1000}.png" for i in range(11)]

    # Load the images
    images = [Image.open(path) for path in image_paths]

    # Titles for each image
    titles = [f"Generation {i*1000}" for i in range(11)]

    # Create a grid of subplots
    num_cols = 4
    num_rows = (len(images) - 1) // num_cols + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Plot each image on a different subplot
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(f"{titles[i]}")
        else:
            # Remove the last subplot if it is empty
            fig.delaxes(ax)

    # Center the last row of subplots if there are an odd number of images
    if len(images) % 2 == 1:
        last_row_axes = axes[-1]
        for ax in last_row_axes:
            bbox = ax.get_position()
            bbox = bbox.translated(0.1, 0.0)
            ax.set_position(bbox)

    plt.savefig(f"results/{folder}/combined_{folder}.png")
    plt.clf()
    plt.close()


subfolder = "results"

# Get all folders names
folders = [
    f for f in os.listdir(subfolder) if os.path.isdir(os.path.join(subfolder, f))
]

folders.sort()
for folder in folders:
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
    print(all[-1])
    plt.plot(all)
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.title(f"Best Fitness for {folder}")
    plt.savefig(f"results/{folder}/plot_{folder}_best_fitness.png")
    plt.clf()
    plt.close()

    all = all[1000:]
    # start from 1000 to the end of length of all
    x = list(range(1000, len(all) + 1000))

    plt.plot(x, all)
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.title(f"Best Fitness for {folder}")
    plt.savefig(f"results/{folder}/plot_{folder}_best_fitness_after_1000.png")
    plt.clf()
    plt.close()

    # Combine images
    combine_images()

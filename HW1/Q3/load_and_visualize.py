# Created by Deniz Karakay at 17.04.2023
# Filename: load_and_visualize.py

import pickle

from utils.utils import part3Plots, visualizeWeights

filename = 'results/question_3_mlp1.pkl'
with open(filename, 'rb') as f:
    loaded_dict = pickle.load(f)

    print(loaded_dict.keys())

    visualizeWeights(loaded_dict['weights'], save_dir='results/',
                     filename='question_3_weights_'+loaded_dict['name'].replace(' ', '_'))

    # part3Plots (

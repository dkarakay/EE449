# Created by Deniz Karakay at 17.04.2023
# Filename: load_and_visualize.py

import pickle

from utils.utils import part3Plots, visualizeWeights

temp = []
for name in ['mlp1', 'mlp2', 'cnn3', 'cnn4', 'cnn5']:
    filename = f'results/{name}/question_3_{name}.pkl'
    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)
        temp.append(loaded_dict)
        print(loaded_dict.keys())

        print(type(loaded_dict['weights']))

part3Plots(temp, save_dir='results/', filename='question_3_plots')

# Created by Deniz Karakay at 18.04.2023
# Filename: load_and_visualize.py
import pickle

from utils.utils import part4Plots

temp = []
for name in ['mlp1', 'mlp2', 'cnn3', 'cnn4', 'cnn5']:
    filename = f'results/question_4_{name}.pkl'
    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)
        temp.append(loaded_dict)
        print(loaded_dict.keys())

part4Plots(temp, save_dir='results/', filename='question_4_plots')

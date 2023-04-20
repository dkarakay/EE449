# Created by Deniz Karakay at 18.04.2023
# Filename: load_and_visualize.py
import pickle

from utils.utils import part5Plots

filename = 'question_5_cnn4_001.pkl'
with open(filename, 'rb') as f:
    loaded_dict = pickle.load(f)
    print(loaded_dict.keys())

    part5Plots(loaded_dict, save_dir='', filename='question_5_plots')

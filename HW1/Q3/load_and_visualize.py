# Created by Deniz Karakay at 17.04.2023
# Filename: load_and_visualize.py

import pickle

from utils.utils import part3Plots, visualizeWeights

temp = []
for name in ['cnn5']:
    filename = f'results/question_3_{name}.pkl'
    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)
        temp.append(loaded_dict)
        print(loaded_dict.keys())

        #loaded_dict['test_acc'] = loaded_dict['test_acc']
        #loaded_dict['val_acc_curve'] = loaded_dict['val_acc_curve'] * 100
        #loaded_dict['train_acc_curve'] = loaded_dict['train_acc_curve'] * 100


part3Plots(temp, save_dir='results/', filename='question_32')
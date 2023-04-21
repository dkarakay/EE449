# Created by Deniz Karakay at 20.04.2023
# Filename: visualize_valid_final.py

import pickle
import numpy as np
from matplotlib import pyplot as plt
import os


# Custom part5Plots function
def custom_plot_for_part_2(result, save_dir='', filename='', show_plot=True):
    if isinstance(result, (list, tuple)):
        result = result[0]

    color_list = ['#dad420', ]
    style_list = ['-', '--']

    num_curves = 1

    plot_args = [{'c': color_list[k],
                  'linestyle': style_list[0],
                  'linewidth': 2} for k in range(num_curves)]

    key_suffixes = ['1', ]

    font_size = 18

    fig, axes = plt.subplots(1, figsize=(16, 12))

    fig.suptitle('training of cnn4 with three different learning rates: 0.1, 0.01, 0.001',
                 fontsize=font_size, y=0.025)

    # training loss and validation accuracy
    axes.set_title('validation_accuracy', loc='right', fontsize=font_size)
    for key_suffix, plot_args in zip(key_suffixes, plot_args):
        acc_curve = result['val_acc_curve']
        label = 'lr=0.1, 0.01 and 0.001'

        axes.plot(np.arange(1, len(acc_curve) + 1),
                  acc_curve, label=label, **plot_args)
        axes.set_xlabel(xlabel='step', fontsize=font_size)
        axes.set_ylabel(ylabel='accuracy', fontsize=font_size)
        axes.tick_params(labelsize=12)

    # global legend
    lines = axes.get_lines()
    fig.legend(labels=[line._label for line in lines],
               ncol=3, loc="upper center", fontsize=font_size,
               handles=lines)

    if show_plot:
        plt.show()

    fig.savefig(os.path.join(save_dir, filename + '.png'))


filename = 'results/question_5_part2_final_cnn4.pkl'
with open(filename, 'rb') as f:
    loaded_dict = pickle.load(f)
    print(loaded_dict.keys())

    custom_plot_for_part_2(loaded_dict, save_dir='', filename='question_5_plots_part2_final')

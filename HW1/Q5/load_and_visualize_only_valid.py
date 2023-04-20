# Created by Deniz Karakay at 20.04.2023
# Filename: load_and_visualize_only_valid.py


import pickle
import numpy as np
from matplotlib import pyplot as plt
import os


# utility function to create performance plots for part 5
def part5Plots(result, save_dir='', filename='', show_plot=True):
    """plots multiple performance curves from multiple training results and
    saves the resultant plot as a png image

    Arguments:
    ----------

    result: dictionary object, each corresponds to
    the result of a training and should have the following key-value
    items:

        'name': string, indicating the user-defined name of the training

        'loss_curve_1': list of floats, indicating the loss with .1 lr at each step

        'loss_curve_01': list of floats, indicating the loss with .01 lr at each step

        'loss_curve_001': list of floats, indicating the loss with .001 lr at each step

        'val_acc_curve_1': list of floats, indicating the val acc with .1 lr at each step

        'val_acc_curve_01': list of floats, indicating the val acc with .01 lr at each step

        'val_acc_curve_001': list of floats, indicating the val acc with .001 lr at each step

    save_dir: string, indicating the path to directory where the plot image is to be saved

    filename: string, indicating the name of the image file. Note that .png will be automatically
    appended to the filename.

    show_plot: bool, whether the figure is to be shown

    Example:
    --------

    visualizing the results of the training

    # assume the '*_value's are known

    >>> result = {'name': name_value, 'loss_curve_1': loss_curve_1_value, ...}

    >>> part4Plots(result, save_dir=r'some\location\to\save', filename='part4Plots')

    """

    if isinstance(result, (list, tuple)):
        result = result[0]

    color_list = ['#0000ff', '#ff0000', '#d2691e', '#ff00ff', '#00ff00', '#000000', '#373788']
    style_list = ['-', '--']

    num_curves = 3

    plot_args = [{'c': color_list[k],
                  'linestyle': style_list[0],
                  'linewidth': 2} for k in range(num_curves)]

    key_suffixes = ['1', '01', '001']

    font_size = 18

    fig, axes = plt.subplots(1, 2, figsize=(16, 12))

    fig.suptitle('training of <%s> with different learning rates' % result['name'],
                 fontsize=font_size, y=0.025)

    # training loss and validation accuracy
    axes[0].set_title('validation_accuracies', loc='right', fontsize=font_size)
    for key_suffix, plot_args in zip(key_suffixes, plot_args):
        acc_curve = result['val_acc_curve_' + key_suffix]
        label = 'lr=0.%s' % key_suffix

        axes[0].plot(np.arange(1, len(acc_curve) + 1),
                     acc_curve, label=label, **plot_args)
        axes[0].set_xlabel(xlabel='step', fontsize=font_size)
        axes[0].set_ylabel(ylabel='accuracy', fontsize=font_size)
        axes[0].tick_params(labelsize=12)

    # global legend
    lines = axes[0].get_lines()
    fig.legend(labels=[line._label for line in lines],
               ncol=3, loc="upper center", fontsize=font_size,
               handles=lines)

    if show_plot:
        plt.show()

    fig.savefig(os.path.join(save_dir, filename + '.png'))


filename = 'question_5_cnn4_001.pkl'
with open(filename, 'rb') as f:
    loaded_dict = pickle.load(f)
    print(loaded_dict.keys())

    part5Plots(loaded_dict, save_dir='', filename='question_5_plots_part2')

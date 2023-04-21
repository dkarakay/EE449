#   This file contains the code for Question 2

import numpy as np
import torch
from utils.utils import part2Plots


def my_conv2d(input_tensor, kernel):
    """
    :param input_tensor: (batch_size, in_channels, in_height, in_width)
    :param kernel: (out_channels, in_channels, kernel_height, kernel_width)
    :return:
    """

    batch_size, i_channels, i_height, i_width = input_tensor.shape
    o_channels_kernel, i_channels_kernel, f_height, f_width = kernel.shape

    print("Input:")
    print(f"Batch size: {batch_size}")
    print(f"Input channels: {i_channels}")
    print(f"Input height: {i_height}")
    print(f"Input width: {i_width}")

    print("------------------")
    print("Kernel:")
    print(f"Output channels: {o_channels_kernel}")
    print(f"Input channels: {i_channels_kernel}")
    print(f"Filter height: {f_height}")
    print(f"Filter width: {f_width}")
    print("------------------")

    # Calculate output dimensions
    o_height = i_height - f_height + 1
    o_width = i_width - f_width + 1

    output = np.zeros((batch_size, o_channels_kernel, o_height, o_width))

    print("Output shape: ", output.shape)

    # 2D convolution
    for b in range(batch_size):  # batch
        for x in range(o_height):  # height
            for y in range(o_width):  # width
                for o in range(o_channels_kernel):  # output channels
                    output[b, o, x, y] = np.sum(
                        input_tensor[b, :, x:x + f_height, y:y + f_width, ] * kernel[o, :, :, :]
                    )

    return output


# Pick based on my Student ID
input_sample = np.load('../utils/samples_7.npy')
kernel = np.load('../utils/kernel.npy')

# Perform 2D convolution with my_conv2d
my_output = my_conv2d(input_sample, kernel)

# Normalize input
my_output_normalized = (my_output - np.min(my_output)) / (np.max(my_output) - np.min(my_output))

# Plot results
part2Plots(out=my_output, save_dir='results/', filename='q2_result')
part2Plots(out=my_output_normalized, save_dir='results/', filename='q2_result_normalized')

# Compare with PyTorch's conv2d function
input_tensor = torch.from_numpy(input_sample).float()
kernel_tensor = torch.from_numpy(kernel).float()
output_torch = torch.conv2d(input_tensor, kernel_tensor, stride=1, padding=0)
output_torch_normalized = (output_torch - torch.min(output_torch)) / (torch.max(output_torch) - torch.min(output_torch))

# Plot results
part2Plots(out=output_torch.detach().numpy(), save_dir='results/', filename='q2_result_torch')
part2Plots(out=output_torch_normalized.detach().numpy(), save_dir='results/', filename='q2_result_normalized_torch')

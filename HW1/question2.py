import numpy as np
import torch

from utils import part2Plots


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

    for x in range(o_height):
        for y in range(o_width):
            # Slice the input_tensor to extract a (kernel_size, kernel_size) window of pixels centered around (x,y)
            input_slice = input_tensor[:, :, x:x + f_height, y:y + f_width]

            print("Input slice shape: ", input_slice.shape)


            print("Kernel shape: ", kernel.shape)

            # Reshape the kernel weight to a 2D tensor of shape (out_channels, in_channels * kernel_size * kernel_size)

            print(i_channels_kernel * f_width * f_height)
            kernel_reshaped = kernel.reshape(o_channels_kernel, (i_channels_kernel * f_width * f_height))

            print("Kernel reshaped shape: ", kernel_reshaped.shape)
            # Compute the product between the kernel and the input tensor slice
            product = input_slice * kernel_reshaped.reshape(1, o_channels_kernel, -1)

            # Sum the elements of the tensor obtained along the last two dimensions (kernel_size, kernel_size).
            conv_sum = np.sum(product, axis=(2, 3))

            # Set the value of the corresponding output pixel
            output[:, :, x, y] = conv_sum

    return output


input_sample = np.load('samples_7.npy')
kernel = np.load('kernel.npy')

my_output = my_conv2d(input_sample, kernel)

part2Plots(out=my_output, save_dir='results/', filename='q2_result')

# Compare with PyTorch's conv2d function
x_torch = torch.from_numpy(input_sample)
weight_torch = torch.from_numpy(kernel)
# output_torch = torch.nn.functional.conv2d(x_torch, weight_torch)

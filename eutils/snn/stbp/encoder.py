#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   encoder.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/2/27 19:40   lintean      1.0         None
'''
from typing import Union, Callable

import torch


def euclidean_distance(x, y):
    """
    Simple euclidean distance metric.
    """
    return (x - y).pow(2)


def gaussian_rbf(tensor: torch.Tensor, sigma: float = 1):
    """
    A `gaussian radial basis kernel <https://en.wikipedia.org/wiki/Radial_basis_function_kernel>`_
    that calculates the radial basis given a distance value (distance between :math:`x` and a data
    value :math:`x'`, or :math:`\\|\\mathbf{x} - \\mathbf{x'}\\|^2` below).

    .. math::
        K(\\mathbf{x}, \\mathbf{x'}) = \\exp\\left(- \\frac{\\|\\mathbf{x} - \\mathbf{x'}\\|^2}{2\\sigma^2}\\right)

    Parameters:
        tensor (torch.Tensor): The tensor containing distance values to convert to radial bases
        sigma (float): The spread of the gaussian distribution. Defaults to 1.
    """
    return torch.exp(-tensor / (2 * sigma ** 2))


class PopulationEncoder(torch.nn.Module):
    """Encodes a set of input values into population codes, such that each singular input value is represented by
    a list of numbers (typically calculated by a radial basis kernel), whose length is equal to the out_features.

    Population encoding can be visualised by imagining a number of neurons in a list, whose activity increases
    if a number gets close to its "receptive field".

    .. figure:: https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/PopulationCode.svg/1920px-PopulationCode.svg.png

        Gaussian curves representing different neuron "receptive fields". Image credit: `Andrew K. Richardson`_.

    .. _Andrew K. Richardson: https://commons.wikimedia.org/wiki/File:PopulationCode.svg

    Example:
        >>> data = torch.as_tensor([0, 0.5, 1])
        >>> out_features = 3
        >>> PopulationEncoder(out_features).forward(data)
        tensor([[1.0000, 0.8825, 0.6065],
                [0.8825, 1.0000, 0.8825],
                [0.6065, 0.8825, 1.0000]])

    Parameters:
        out_features (int): The number of output *per* input value
        scale (torch.Tensor): The scaling factor for the kernels. Defaults to the maximum value of the input.
                            Can also be set for each individual sample.
        kernel: A function that takes two inputs and returns a tensor. The two inputs represent the center value
                (which changes for each index in the output tensor) and the actual data value to encode respectively.z
                Defaults to gaussian radial basis kernel function.
        distance_function: A function that calculates the distance between two numbers. Defaults to euclidean.
    """

    def __init__(
        self,
        out_features: int,
        scale: Union[int, torch.Tensor] = None,
        kernel: Callable[[torch.Tensor], torch.Tensor] = gaussian_rbf,
        distance_function: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = euclidean_distance,
    ):
        super(PopulationEncoder, self).__init__()
        self.out_features = out_features
        self.scale = scale
        self.kernel = kernel
        self.distance_function = distance_function

    def forward(self, input_tensor):
        return population_encode(
            input_tensor,
            self.out_features,
            self.scale,
            self.kernel,
            self.distance_function,
        )


def population_encode(
    input_values: torch.Tensor,
    out_features: int,
    scale: Union[int, torch.Tensor] = None,
    kernel: Callable[[torch.Tensor], torch.Tensor] = gaussian_rbf,
    distance_function: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = euclidean_distance,
) -> torch.Tensor:
    """
    Encodes a set of input values into population codes, such that each singular input value is represented by
    a list of numbers (typically calculated by a radial basis kernel), whose length is equal to the out_features.

    Population encoding can be visualised by imagining a number of neurons in a list, whose activity increases
    if a number gets close to its "receptive field".

    .. figure:: https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/PopulationCode.svg/1920px-PopulationCode.svg.png

        Gaussian curves representing different neuron "receptive fields". Image credit: `Andrew K. Richardson`_.

    .. _Andrew K. Richardson: https://commons.wikimedia.org/wiki/File:PopulationCode.svg

    Example:
        >>> data = torch.as_tensor([0, 0.5, 1])
        >>> out_features = 3
        >>> pop_encoded = population_encode(data, out_features)
        tensor([[1.0000, 0.8825, 0.6065],
                [0.8825, 1.0000, 0.8825],
                [0.6065, 0.8825, 1.0000]])
        >>> spikes = poisson_encode(pop_encoded, 1).squeeze() # Convert to spikes

    Parameters:
        input_values (torch.Tensor): The input data as numerical values to be encoded to population codes
        out_features (int): The number of output *per* input value
        scale (torch.Tensor): The scaling factor for the kernels. Defaults to the maximum value of the input.
                              Can also be set for each individual sample.
        kernel: A function that takes two inputs and returns a tensor. The two inputs represent the center value
                (which changes for each index in the output tensor) and the actual data value to encode respectively.z
                Defaults to gaussian radial basis kernel function.
        distance_function: A function that calculates the distance between two numbers. Defaults to euclidean.

    Returns:
        A tensor with an extra dimension of size `seq_length` containing population encoded values of the input stimulus.
        Note: An extra step is required to convert the values to spikes, see above.
    """
    # Thanks to: https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py
    size = (input_values.size(0), out_features) + input_values.size()[1:]
    if not scale:
        scale = input_values.max()
    centres = torch.linspace(0, scale, out_features).expand(size)
    x = input_values.unsqueeze(1).expand(size)
    distances = distance_function(x, centres) * scale
    return kernel(distances)


"""Provides function for cloning pytorch modules"""
import copy
from torch import nn


def clones(module: nn.Module, num_clones: int) -> nn.ModuleList:
    """
    Create a list of cloned modules.

    Args:
        module (nn.Module): The module to be cloned.
        num_clones (int): The number of clones to create.

    Returns:
        nn.ModuleList: A list of cloned modules.

    This function creates a list of cloned modules by deep copying the provided module the specified
    number of times.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_clones)])

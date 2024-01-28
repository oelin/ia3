from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IA3(nn.Module):
    """IA3.

    IA3 is a PEFT technique that scales the activations of particular modules by
    learned vectors.

    Example
    -------
    >>> module = IA3(
    ...     module=...,
    ...     embedding_dimension=256,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> module.enable_ia3()
    >>> x = module(x)
    >>> module.disable_ia3()
    >>> x = module(x)
    """

    def __init__(
        self, 
        *,
        module: nn.Module,
        embedding_dimension: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        module : nn.Module
            The module to adapt.
        embedding_dimension : int
            The embedding dimension of the module.
        """

        super().__init__()

        self.module = module
        self.scale = nn.Parameter(torch.ones((embedding_dimension,)))
        self.is_enabled = False
    
    def enable_ia3(self) -> None:
        """Enable IA3."""

        self.is_enabled = True
    
    def disable_ia3(self) -> None:
        """Disable IA3."""

        self.is_enabled = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.
        
        Parameter
        ---------
        x : torch.Tensor
            The input tensor.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        if self.is_enabled:
            return self.module(x) * self.scale
        
        return self.module(x)

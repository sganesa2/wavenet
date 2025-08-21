import torch
from typing import Iterator, OrderedDict, ClassVar

from model.layers import (
    Module, FlattenConsecutive, Linear, BatchNorm1d, Tanh
)

class Sequential(Module):
    def __init__(self, *args)->None:
        self._modules: dict[str, Module] = OrderedDict()

        for idx, module in enumerate(args):
            if isinstance(module, Module):
                self._modules[idx] = module

    def __iter__(self)->Iterator[Module]:
        return iter(self._modules.values())
    
    def __len__(self)->int:
        return len(self._modules)
    
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x
    
    @property
    def params(self)->Iterator[torch.Tensor]:
        params = []
        for layer in self._modules.values():
            params.extend(layer.params)
        return iter(params)
    
    def append(self, module:Module)->None:
        if not isinstance(module, Module):
            raise TypeError("You have passed a module that isn't of type Module")
        
        self._modules[len(self)] = module
    
    def insert(self, idx:int, module:Module)->None:
        if not isinstance(module, Module):
            raise TypeError("You have passed a module that isn't of type Module")
        
        n = len(self._modules)
        if not (-n <= idx <= n):
            raise IndexError(f"Index out of range: {idx}")
        if idx<0:
            idx +=n

        for i in range(n, idx, -1):
            self._modules[i] = self._modules[i-1]
        self._modules[idx] = module


class SimpleDilatedConvolution(Module):
    def __init__(self, dilation_factor:int, h:int, feature_dims:int)->None:
        """
        Building block of wavenet that is composed of:
            - <FlattenConsecutive>, <Linear>, <BatchnNorm1d>, <Tanh>
        """
        self.dilation_factor = dilation_factor
        self.h = h
        self.feature_dims = feature_dims

        self.dilated_convolution_block = Sequential(
            FlattenConsecutive(dilation_factor),
            Linear(dilation_factor*feature_dims, h, bias = False).with_kaiming_init('tanh'),
            BatchNorm1d(h),
            Tanh()
        )
    def __iter__(self)->Iterator[Module]:
        return iter(self.dilated_convolution_block)
    
    def __len__(self)->int:
        return len(self.dilated_convolution_block)
    
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        return self.dilated_convoulution_block(x)
    
    @property
    def params(self)->Iterator[torch.Tensor]:
        return self.dilated_convolution_block.params
    
    @classmethod
    def recursive_convolution_init(cls, n:int, dilation_factor:int, h:int, feature_dims:int)->Iterator['SimpleDilatedConvolution']:
        convolutions = []
        def _recursive_int(n:int)->'SimpleDilatedConvolution':
            if n==1: return
            nonlocal convolutions, dilation_factor, h, feature_dims
            convolutions.append(
                cls(dilation_factor, h, feature_dims)
            )
            _recursive_int(n//dilation_factor)
        _recursive_int(n)
        return iter(convolutions)
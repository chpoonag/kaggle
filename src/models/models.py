import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, dims, activation=nn.ReLU(), skip_connection=False):
        super(MLP, self).__init__()
        layers = []
        skip_connection_layers = []
        for h, (i, j) in enumerate(dims):
            layers.append(nn.Linear(i, j))
#             if activation is not None:
#                 layers.append(activation)
                
            if skip_connection and len(dims)>1 and h<len(dims)-1:
                skip_connection_layers.append(nn.Linear(i, dims[-1][-1]))
#                 if activation is not None:
#                     skip_connection_layers.append(activation)
        
        skip_connection = skip_connection and len(skip_connection_layers)>0
        self.layers = nn.ModuleList(layers)
        self.activation_layer = activation
        self.skip_connection_layers = nn.ModuleList(skip_connection_layers) if skip_connection else None
        self._skip_connection = skip_connection

    def forward(self, x, **kwargs):
        xs_skip = []
        for i, layer in enumerate(self.layers):
            if self._skip_connection and i<len(self.skip_connection_layers):
                _skip_connection_layer = self.skip_connection_layers[i]
                _x_skip = self.activation_layer(_skip_connection_layer(x))
                xs_skip.append(_x_skip)
            
            x = layer(x)
            x = self.activation_layer(x)
        if self._skip_connection:
            x = torch.sum(torch.stack([x, *xs_skip]), dim=0)
        return x
    
    def reset_parameters(self):
        for layer in self.layers:
            try:
                layer.reset_parameters()
            except AttributeError:    # nn.ReLU() has no reset_parameter method, due to no learnable parameters
                pass
        if self._skip_connection:
            for layer in self.skip_connection_layers:
                try:
                    layer.reset_parameters()
                except AttributeError:    # nn.ReLU() has no reset_parameter method, due to no learnable parameters
                    pass
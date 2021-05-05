import torch
import torch.nn.functional as F
import torch.nn as nn
#from utils import get_layer_mappings,get_activation_mappings

class Layer:
    def __init__(self,layer_dict):
        self.layer_dict = layer_dict
        self.layer_type = layer_dict['layer_type']
        self.layer_name = layer_dict['layer_name']
        self.units_in = layer_dict['units_in']
        self.units_out = layer_dict['units_out']
        self.activation = layer_dict['activation']
        self.layer_mappings = {'dense':nn.Linear}
        self.activation_mappings = {'relu':nn.ReLU}

    def get_torch_layer(self):
        layer = self.layer_mappings[self.layer_type](in_features=self.units_in,
        out_features = self.units_out)
        activation = self.activation_mappings[self.activation]()

        return (self.layer_name,layer),(self.layer_name+'_act',activation)
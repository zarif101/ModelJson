import torch
import torch.nn as nn
from layers import Layer
import collections
from collections import OrderedDict

def get_layer_mappings():
    map = {'dense':nn.Linear}
    return map
def get_activation_mappings():
    map = {'relu':nn.ReLU}
    return map

def get_layers_dict(json_list):
    layers_list = []
    for layer_info in json_list:
        #(layer_name,layer),(layer_activation_name,layer_activation) = Layer(layer_info)
        layer = Layer(layer_info)
        layer_tup,activation_tup = layer.get_torch_layer()
        layers_list.append(layer_tup)
        layers_list.append(activation_tup)
    #ret = dict([(item[0], item[1]) for item in layers_list])
    ret = collections.OrderedDict(layers_list)
    return ret

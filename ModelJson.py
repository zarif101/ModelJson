import json
import argparse
from utils import get_layers_dict
import torch

def get_model(jsonpath):
    with open(jsonpath) as f:
        data = json.load(f)
    layers_dict = get_layers_dict(data)
    model = torch.nn.Sequential(layers_dict)
    return model

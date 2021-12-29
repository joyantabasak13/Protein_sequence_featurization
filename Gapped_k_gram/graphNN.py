import numpy as np
import math
import json
import pickle
import networkx as nx
import itertools

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import dgl
from dgl.nn.pytorch import GraphConv

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from sklearn import manifold


def load_list_of_dicts(filename, create_using=nx.MultiGraph):

    with open(filename, 'rb') as f:
        list_of_dicts = pickle.load(f)

    graphs = [create_using(graph) for graph in list_of_dicts]

    return graphs


graph_file_name = 'kmer_graphs.pkl'
graphs = load_list_of_dicts(graph_file_name)

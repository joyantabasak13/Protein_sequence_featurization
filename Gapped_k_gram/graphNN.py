import numpy as np
import pandas as pd
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


# We want to experiment with the amount of computation required in the
# forward pass, so we'll build various GNNs here, each with different
# numbers of transition function layers
class StargazerGNN(nn.Module):
    def __init__(self, num_hidden_features):
        super().__init__()

        # We'll apply some number of spatial convolutions / message passing
        # RGNNs
        self.convs = nn.ModuleList()
        for i in range(len(num_hidden_features) - 1):
            self.convs.append(GraphConv(num_hidden_features[i],
                                        num_hidden_features[i + 1]))

        # Classify out to one of two classes
        self.classify = nn.Linear(num_hidden_features[-1], 2)

    def forward(self, g):
        # Start with just the degree as a feature
        h = g.in_degrees().view(-1, 1).float()
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
            h = torch.relu(h)

        # Calculate graph representation by averaging all the node
        # representations, thus making a graph representation
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        output = self.classify(hg)
        return output

    def hidden(self, g):
        # Start with just the degree as a feature
        h = g.in_degrees().view(-1, 1).float()
        for i in range(len(self.convs)):
            h = self.convs[i](g, h)
            h = torch.relu(h)

        return h


#Loads pickled graphs
def load_list_of_dicts(filename, create_using=nx.MultiDiGraph):

    with open(filename, 'rb') as f:
        list_of_dicts = pickle.load(f)

    graphs = [create_using(graph) for graph in list_of_dicts]

    return graphs


# Helper for getting batches from a dataset
def get_batches(xs, ys, batch_size=16):
    # How many batches is there of given size for this dataset
    num = len(xs)
    num_batches = math.ceil(num / batch_size)

    # Go through and get all batches
    batches_x = []
    batches_y = []

    # Get all batches in memory
    for i in range(num_batches - 1):
        sidx = batch_size * i
        fidx = batch_size * (i + 1)
        fidx = min(fidx, num)
        batches_x.append(xs[sidx:fidx])
        batches_y.append(ys[sidx:fidx])

    return batches_x, batches_y


graph_file_name = 'kmer_graphs.pkl'
nx_graphs = load_list_of_dicts(graph_file_name)

df = pd.read_csv(r'../Data/miniDataset_window_15 .csv')
labels = ["nonsuc", "suc"]

dgl_graphs_x = []
dgl_graphs_y = df['Class'].apply(labels.index)

#Convert to DGL graphs
for nx_graph in nx_graphs:
    g = dgl.from_networkx(nx_graph, edge_attrs=['weight'])
    dgl_graphs_x.append(g)


# The following operations will train the GNN
def train_gapped_kmer_gnn(gnn, num_epochs):
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(itertools.chain(gnn.parameters()),
                           lr=0.001)

    # Get the batches to work with
    num_train = 360
    batches_x, batches_y = get_batches(dgl_graphs_x[:num_train],
                                       dgl_graphs_y[:num_train],
                                       batch_size=10)

    # Run every batch in every epoch
    epoch_losses = []
    for epoch_index in range(num_epochs):
        epoch_loss = 0
        for batch_index in range(len(batches_x)):
            # Get the batch of interest
            batch_x = batches_x[batch_index]
            batch_y = batches_y[batch_index]

            # Calculate an output for each graph
            x = dgl.batch(batch_x)
            y_hat = gnn(x)
            # And compare to the true value
            y = torch.tensor(batch_y.to_numpy())
            # print(y_hat)
            # print(y)

            # Calculate loss
            loss = loss_func(y_hat, y)

            # Calculate loss and perform gradient descent step accordingly
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
        epoch_loss /= len(batches_x)
        print(f"Epoch {epoch_index}: loss {epoch_loss}")
        epoch_losses.append(epoch_loss)

    # Return the trained architecture and loss
    return epoch_losses


# The following operation will test the GNN, returning hidden representations and
# predictions
def test_gapped_kmer_gnn(gnn):
    num_train = 360
    num_test = 40

    batches_x, batches_y = get_batches(dgl_graphs_x[num_train:num_train + num_test],
                                       dgl_graphs_y[num_train:num_train + num_test],
                                       batch_size=1)

    num_correct = 0
    y_true = []
    y_pred = []
    hidden = []
    for batch_index in range(len(batches_x)):
        # Get the batch of interest
        batch_x = batches_x[batch_index]
        batch_y = batches_y[batch_index]

        # Calculate an output for each graph
        x = dgl.batch(batch_x)
        y_hat = gnn(x)
        y_hat = torch.argmax(y_hat, dim=1).detach().numpy()

        # Also get some hidden representations for plotting
        h = torch.mean(gnn.hidden(x), dim=0).detach().numpy()
        hidden.append(h)

        # Take note of everything
        y_pred.extend(y_hat)
        y_true.extend(batch_y)
    hidden = np.array(hidden)

    return hidden, y_true, y_pred


def train_test_architecture(num_hidden_features, num_epochs, plot_loss=False):
    # Make architecture and train
    gnn_run = StargazerGNN(num_hidden_features)
    epoch_losses = train_gapped_kmer_gnn(gnn_run, num_epochs)

    if num_epochs > 0 and plot_loss:
        plt.plot(range(num_epochs), epoch_losses)
        plt.ylim((0, 1.0))
        plt.title("Loss during training")
        plt.show()

    # Test and return hidden representations plus predictions
    hidden, y_true, y_pred = test_gapped_kmer_gnn(gnn_run)

    # Print performance metrics
    print(metrics.classification_report(y_true, y_pred,
                                        target_names=['nonsuc', 'suc']))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    print(f"AUC\t{metrics.auc(fpr, tpr)}")

    # Plot TSNE visualization
    tsne = manifold.TSNE(n_components=2)
    tsne_embedded = tsne.fit_transform(hidden)
    print(tsne_embedded.shape)
    plt.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1],
                marker='.', c=y_true,
                cmap=matplotlib.colors.ListedColormap(['purple', 'green']))


train_test_architecture([1, 32, 32, 32, 32], 6)

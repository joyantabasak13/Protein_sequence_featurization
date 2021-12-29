import numpy as np
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def store_as_list_of_dicts(filename, graphs):

    list_of_dicts = [nx.to_dict_of_dicts(graph) for graph in graphs]

    with open(filename, 'wb') as f:
        pickle.dump(list_of_dicts, f)


df = pd.read_csv(r'../Data/miniDataset_window_15 .csv')

num_rows = len(df.index)
sequence_length = len(df['Sequence'][0])

kmer_len = 3

graph_list = []
for seq in df['Sequence']:
    seq_graph = nx.MultiGraph()
    print(seq)
    for i in range(sequence_length - kmer_len):
        kmer_seq_i = seq[i: i + kmer_len]
        if not seq_graph.has_node(kmer_seq_i):
            seq_graph.add_node(kmer_seq_i)
        for j in range(i + 1, sequence_length - kmer_len):
            kmer_seq_j = seq[j: j + kmer_len]
            if not seq_graph.has_node(kmer_seq_j):
                seq_graph.add_node(kmer_seq_i)
            edge_weight = 1 / abs(j - i)
            seq_graph.add_edge(kmer_seq_i, kmer_seq_j, weight=edge_weight)
    print(seq_graph)
    graph_list.append(seq_graph)
    # pos = nx.spring_layout(seq_graph)
    # nx.draw_networkx_nodes(seq_graph, pos, cmap=plt.get_cmap('jet'), node_size=100)
    # nx.draw_networkx_edges(seq_graph, pos, edge_color='r', arrows=True)
    # nx.draw_networkx_labels(seq_graph, pos)
    # plt.show()
    # exit()


graph_file_name = 'kmer_graphs.pkl'
store_as_list_of_dicts(graph_file_name, graph_list)
print("Saving Done")


def load_list_of_dicts(filename, create_using=nx.MultiGraph):

    with open(filename, 'rb') as f:
        list_of_dicts = pickle.load(f)

    graphs = [create_using(graph) for graph in list_of_dicts]

    return graphs

graphs = load_list_of_dicts(graph_file_name)
for graph in graphs:
    print(graph)

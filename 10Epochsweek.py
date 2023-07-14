import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding
import networkx as nx

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar,StellarGraph
from stellargraph import datasets
from IPython.display import display, HTML
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML
import pandas as pd
import numpy as np

g1 = nx.read_graphml("2019-12-31.graphml")
g2 = nx.read_graphml("2020-01-01.graphml")
g3 = nx.read_graphml("2020-01-02.graphml")
g4 = nx.read_graphml("2020-01-03.graphml")
g5 = nx.read_graphml("2020-01-04.graphml")
g6 = nx.read_graphml("2020-01-05.graphml")
g7 = nx.read_graphml("2020-01-06.graphml")
g8 = nx.read_graphml("2020-01-07.graphml")

glist = [g2,g3,g4,g5,g6,g7,g8]
idx = len(glist)

G = nx.Graph()
while idx >= 0:
    idx -= 1
    G = nx.compose(G, glist[idx])
    attr_n_tx = {e: G.edges[e]['n_tx'] + glist[idx].edges[e]['n_tx'] for e in G.edges & glist[idx].edges}
    nx.set_edge_attributes(G, attr_n_tx, 'n_tx')
    attr_value = {e: G.edges[e]['value'] + glist[idx].edges[e]['value'] for e in G.edges & glist[idx].edges}
    nx.set_edge_attributes(G, attr_value, 'value')

def max_min_normalization(G):
    n_tx = nx.get_edge_attributes(G, "n_tx")
    value = nx.get_edge_attributes(G, "value")
    weight = {}
    n_tx_min = min(n_tx.values())
    n_tx_max = max(n_tx.values())
    value_min = min(value.values())
    value_max = max(value.values())
    for key in n_tx.keys():
        n_tx[key] = (n_tx[key] - n_tx_min)/(n_tx_max - n_tx_min)
    for key in value.keys():
        value[key] = (value[key] - value_min)/(value_max - value_min)
    for key in value.keys():
        weight[key] = n_tx[key] * 0.5 + value[key] * 0.5
    weight_min = min(weight.values())
    weight_max = max(weight.values())
    for key in weight.keys():
        weight[key] = (weight[key] - weight_min)/(weight_max - weight_min)
    nx.set_edge_attributes(G, weight, name="weight")
    return weight

G2 = G.copy()

G2.remove_nodes_from(list(n for n in G2 if n not in g1))

G.remove_nodes_from(list(n for n in G if n not in g1))
G.remove_edges_from(list(n for n in G if n not in g1))

df_nodes2 = nx.to_pandas_adjacency(G)
df_nodes1 = nx.to_pandas_adjacency(G2)

max_min_normalization(G2)
max_min_normalization(G)

Gs1 = StellarGraph.from_networkx(
    G,node_features=df_nodes1
)
Gs2 = StellarGraph.from_networkx(
    G2,node_features=df_nodes2
)

print(Gs1.info())
print(Gs2.info())

# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(Gs2)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=False
)

# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(Gs1)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=False
)

batch_size = 10
epochs = 10
num_samples = [10, 5]

train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

layer_sizes = [10, 10]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
)

x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="relu", edge_embedding_method="ip"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"],
)

init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)

sg.utils.plot_history(history)

train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
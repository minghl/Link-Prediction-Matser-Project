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

g1 = nx.read_graphml("/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2019-12-31.graphml.bz2")
names = locals()
graphs = []
for i in range(4):
    if i == 0:
        for j in range(9):
            names['g' + str(i) +  str(j+1) ] = nx.read_graphml("/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2020-01-"+ str(i) +str(j+1)+".graphml.bz2")
            graphs.append(names['g' + str(i) +  str(j+1) ])
    if i == 1:
        for j in range(10):
            names['g' + str(i) +  str(j+1) ] = nx.read_graphml("/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2020-01-"+str(i)+ str(j)+".graphml.bz2")
            graphs.append(names['g' + str(i) +  str(j+1) ])
    if i == 2:
        for j in range(10):
            names['g' + str(i) +  str(j+1) ] = nx.read_graphml("/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2020-01-"+str(i)+ str(j)+".graphml.bz2")
            graphs.append(names['g' + str(i) +  str(j+1) ])
    if i == 3:
        for j in range(2):
            names['g' + str(i) +  str(j+1) ] = nx.read_graphml("/srv/abacus-1/txnetworks_blockchains/bitcoin/heur_1_networks_day/2020-01-"+str(i)+ str(j)+".graphml.bz2")
            graphs.append(names['g' + str(i) +  str(j+1) ])

idx = len(graphs)

G = nx.Graph()
while idx >= 0:
    idx -= 1
    G = nx.compose(G, graphs[idx])
    attr_n_tx = {e: G.edges[e]['n_tx'] + graphs[idx].edges[e]['n_tx'] for e in G.edges & graphs[idx].edges}
    nx.set_edge_attributes(G, attr_n_tx, 'n_tx')
    attr_value = {e: G.edges[e]['value'] + graphs[idx].edges[e]['value'] for e in G.edges & graphs[idx].edges}
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


G.remove_nodes_from(list(n for n in G if n not in g1))

max_min_normalization(G)

dc1 = nx.degree_centrality(G)
ec1 = nx.eigenvector_centrality(G,weight='weight')
cc1 = nx.closeness_centrality(G)

#df_nodes1 = nx.to_pandas_adjacency(G2)
def compute_features1(node_id):
    # in general this could compute something based on other features, but for this example,
    # we don't have any other features, so we'll just do something basic with the node_id
    return [dc1[node_id],ec1[node_id],cc1[node_id]]



for node_id, node_data in G.nodes(data=True):
    node_data["feature"] = compute_features1(node_id)


Gs = StellarGraph.from_networkx(
    G,node_features="feature"
)

print(Gs.info())

# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(Gs)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=False
)

# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=False
)

batch_size = 20
epochs = 20
num_samples = [20, 10]

train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

layer_sizes = [20, 20]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
)

# Build the model and expose input and output sockets of graphsage model
# for link prediction
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

gra = sg.utils.plot_history(history, return_figure=True)
gra.savefig('Minghao_Month.png')


train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
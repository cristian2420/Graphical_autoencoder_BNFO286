## Project BNFO 286
# Author: Cristian Gonzalez-Colin
# Date: Marrch 12, 2025
###
# Script to find communities within a network using Graph Autoencoders
# The script reads a network from a file, trains a graph autoencoder, 
# and applies density-based clustering to the embeddings. 
# Modification: Use modularity matrix as input
####################
# Import the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import networkx as nx
import numpy as np
import pandas as pd

from torch_geometric.nn import GraphConv
from torch_geometric.data import Data


# Define a Graph Autoencoder
class GAE_GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAE_GraphConv, self).__init__()
        self.encoder = GraphConv(in_channels, out_channels)
        self.decoder = torch.nn.Linear(out_channels, in_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        z = self.encoder(x, edge_index, edge_weight)
        z = torch.relu(z)
        return self.decoder(z)
    
### Create graph data
# Read list of edges from file
network_data = pd.read_csv('network_with_weights.edgelist', sep='\t', header=None)
network_data.columns = ['source', 'target', 'weight']
# file to networkx graph
G = nx.from_pandas_edgelist(network_data, source='source', target='target', edge_attr='weight')
# Calculate the modularity matrix
modularity_matrix = nx.modularity_matrix(G, weight='weight')
# Create a mapping from node labels to numeric IDs
node_mapping = {node: idx for idx, node in enumerate(set(network_data['source']).union(set(network_data['target'])))}

# Convert the edge list into numeric indices
edge_index = torch.tensor([
    [node_mapping[node] for node in network_data['source']],  # Source nodes
    [node_mapping[node] for node in network_data['target']]   # Target nodes
], dtype=torch.long)

# Geometric data
edge_weights = torch.tensor( np.array(network_data['weight'].values), dtype=torch.float)

# Create a graph data object
#x=torch.eye(G.number_of_nodes())
# feed the modularity matrix as input
x = torch.tensor(modularity_matrix, dtype=torch.float)
data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights)

# Create the autoencoder model
in_channels = G.number_of_nodes()
out_channels = 200
model = GAE_GraphConv(in_channels=in_channels, out_channels=out_channels)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    # Forward pass
    output = model(data)
    # Compute the loss
    loss = criterion(output, data.x)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    
# Save the model
torch.save(model.state_dict(), 'model_modmtx.pth')

# Get the embeddings
model.eval()
node_embeddings = model.encoder(data.x, data.edge_index, data.edge_weight)

# Save the embeddings to a file
np.savetxt('node_embeddings_modmtx.csv', node_embeddings.detach().numpy(), delimiter=',')

# Apply density-based clustering to the embeddings
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.5, min_samples=3).fit(node_embeddings.detach().numpy())
labels = clustering.labels_ + 1 # +1 allows to assign all noise points to cluster 0

# Save the cluster labels & node IDs to a file
node_ids = list(node_mapping.keys())
node_labels = pd.DataFrame({'node_id': node_ids, 'cluster_DBSCAN': labels})

# Print the number of clusters
print(f'Number of clusters: {len(set(labels))}')
# Print the number of nodes in each cluster
print(node_labels['cluster_DBSCAN'].value_counts())


from sklearn.cluster import AffinityPropagation
affinityprop = AffinityPropagation().fit(node_embeddings.detach().numpy())
labels_affinity = affinityprop.labels_
node_labels['cluster_affinity'] = labels_affinity
print(f'Number of clusters: {len(set(labels_affinity))}')
print(pd.Series(labels_affinity).value_counts())

from sklearn.cluster import HDBSCAN
hdbscan = HDBSCAN(min_samples=3, cluster_selection_epsilon = 0.05).fit(node_embeddings.detach().numpy())
labels_hdbscan = hdbscan.labels_ + 1 # +1 allows to assign all noise points to cluster 0
node_labels['cluster_hdbscan'] = labels_hdbscan
print(f'Number of clusters: {len(set(labels_hdbscan))}')
print(pd.Series(labels_hdbscan).value_counts())

# format cluster labels with prefix clust 
node_labels['cluster_DBSCAN'] = 'clust' + node_labels['cluster_DBSCAN'].astype(str)
node_labels['cluster_affinity'] = 'clust' + node_labels['cluster_affinity'].astype(str)
node_labels['cluster_hdbscan'] = 'clust' + node_labels['cluster_hdbscan'].astype(str)
# node_labels.to_csv('node_clusters.csv', index=True)

# Plot the clustering results and labels
import matplotlib.pyplot as plt

# Create a dataframe with the node embeddings
df = pd.DataFrame(node_embeddings.detach().numpy())
df['cluster'] = labels_affinity

# Plot the embeddings and save it
plt.figure(figsize=(10, 10))
plt.scatter(df[0], df[1], c=df['cluster'], cmap='tab20')
plt.savefig('clustering_plot_mod.png')

# Plot network G with cluster labels
for node in G.nodes():
    G.nodes[node]['cluster'] = node_labels.loc[node_labels['node_id'] == node, 'cluster_affinity'].values[0]

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=labels_affinity, cmap='tab20', with_labels=True)
plt.savefig('network_clusters_mod.png')



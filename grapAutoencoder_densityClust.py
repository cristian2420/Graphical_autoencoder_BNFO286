## Project BNFO 286
# Author: Cristian Gonzalez-Colin
# Date: Marrch 12, 2025
###
# Script to find communities within a network using Graph Autoencoders
# The script reads a network from a file, trains a graph autoencoder, 
# and applies density-based clustering to the embeddings. 
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
# file to networkx graph
G = nx.from_pandas_edgelist(network_data, source=0, target=1, edge_attr=2)

# Create a mapping from node labels to numeric IDs
node_mapping = {node: idx for idx, node in enumerate(set(network_data[0]).union(set(network_data[1])))}

# Convert the edge list into numeric indices
edge_index = torch.tensor([
    [node_mapping[node] for node in network_data[0]],  # Source nodes
    [node_mapping[node] for node in network_data[1]]   # Target nodes
], dtype=torch.long)

# Geometric data
edge_weights = torch.tensor( np.array(network_data[2].values), dtype=torch.float)

# Create a graph data object
data = Data(x=torch.eye(G.number_of_nodes()), edge_index=edge_index, edge_weight=edge_weights)

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
torch.save(model.state_dict(), 'model.pth')

# Get the embeddings
model.eval()
node_embeddings = model.encoder(data.x, data.edge_index, data.edge_weight)

# Save the embeddings to a file
np.savetxt('node_embeddings.csv', node_embeddings.detach().numpy(), delimiter=',')

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
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Create a dataframe with the node embeddings
# df = pd.DataFrame(node_embeddings.detach().numpy())
# df['cluster'] = labels

# # Plot the clusters
# sns.pairplot(df, hue='cluster')
# plt.show()





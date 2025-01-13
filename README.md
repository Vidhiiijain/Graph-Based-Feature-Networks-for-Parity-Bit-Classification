# Graph-Based-Feature-Networks-for-Parity-Bit-Classification
This project involves the generation and analysis of graph-based feature networks derived from parity bit formulas for a binary classification task.

# GNN Training-
This codebook trains a GNN model to predict whether an edge exists between a pair of nodes in a graph. The model is based on Graph Convolutional Networks (GCN), a popular architecture for graph-based learning tasks.

Key Features:
1. Graph Representation: The input graph is represented using PyTorch Geometric's Data format, where each graph consists of:
- Node Features (x): Each node is represented by a feature vector.
- Edge Connections (edge_index): A sparse matrix indicating the connections between nodes in the graph.
- Edge Labels (y): A label for each edge, indicating whether it is a real or negative edge.
- Edge Classification Model: The model is a Graph Neural Network using two layers of GCN and a fully connected layer for edge prediction.
2. Node Feature Processing: The node features are passed through two GCN layers to learn node representations.
3. Edge Prediction: After feature propagation through the graph, the edge features (concatenation of source and destination node features) are used to predict the edge classification.
4. Batch Processing: The code handles batching of graphs for efficient training using the PyTorch DataLoader.
5. Loss Function: The Binary Cross-Entropy loss (BCEWithLogitsLoss) is used for edge classification.

Dataset-
The dataset consists of multiple graphs stored in .pt files, where each file contains:
- x: Node features tensor (size [num_nodes, num_features]).
- edge_index: Edge indices tensor (size [2, num_edges]).
- y: Edge labels tensor (size [num_edges]), where each label is either 1 (real edge) or 0 (negative sample).

Requirements- torch, torch_geometric, torchvision, matplotlib, numpy

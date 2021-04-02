"""
Pytorch model
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self, edges_dim, nodes_dim, edges_hidden_dim=32, nodes_hidden_dim=256):
        super(Net, self).__init__()
        self.edges_dim = edges_dim # 12
        self.nodes_dim = nodes_dim # 160
        
        self.nodes_linear = nn.Sequential(
            nn.Linear(self.nodes_dim,  nodes_hidden_dim),
            nn.ReLU(),
            nn.Linear(nodes_hidden_dim,  nodes_hidden_dim//2),
            nn.ReLU(),
            nn.Linear(nodes_hidden_dim//2,  nodes_hidden_dim//4)
        )
        self.edges_linear = nn.Sequential(
            nn.Linear(self.edges_dim, edges_hidden_dim),
            nn.ReLU(),
            nn.Linear(edges_hidden_dim,  edges_hidden_dim),
            nn.ReLU(),
            nn.Linear(edges_hidden_dim,  3),
            nn.Tanh()
        )
        self.final_layer = nn.Linear(4,1)

        self.cs = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        x_edges = self.edges_linear(x[:,:self.edges_dim]).view(-1,3)

        x_node1 = x[:,self.edges_dim:self.edges_dim+self.nodes_dim]
        x_node2 = x[:,self.edges_dim+self.nodes_dim:self.edges_dim+2*self.nodes_dim]

        x_node1 = self.nodes_linear(x_node1)
        x_node2 = self. nodes_linear(x_node2)
        nodes_similarity = self.cs(x_node1, x_node2).view(-1,1)

        x = torch.cat((x_edges, nodes_similarity), dim=1)
        return torch.sigmoid(self.final_layer(x)).squeeze()
    
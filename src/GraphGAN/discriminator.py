import torch
import config


class Discriminator(object):
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        self.embedding_matrix = torch.randn(self.node_emd_init.shape)
        self.bias_vector = torch.zeros([self.n_node])

        self.node_id = 0
        self.node_neighbor_id = 0
        self.reward = 0

        self.node_embedding = torch.index_select(self.embedding_matrix, 0, self.node_id.long())
        self.node_neighbor_embedding = torch.index_select(self.embedding_matrix, 0, self.node_neighbor_id.long())
        self.bias = torch.index_select(self.bias_vector, 0, self.node_neighbor_id.long())
        self.score = self.node_embedding*self.node_neighbor_embedding.sum(0) + self.bias

        
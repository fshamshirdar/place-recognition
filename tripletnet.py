import torch
import torch.nn as nn
import torch.nn.functional as F
from l2normalize import L2Normalize

class TripletNet(nn.Module):
    def __init__(self, embeddingnet):
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet
        self.l2norm = L2Normalize()

    def forward(self, x, y, z):
        embedded_x = self.l2norm(self.embeddingnet.features_extraction(x))
        embedded_y = self.l2norm(self.embeddingnet.features_extraction(y))
        embedded_z = self.l2norm(self.embeddingnet.features_extraction(z))
#        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
#        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z

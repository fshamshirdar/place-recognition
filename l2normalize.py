import torch
import torch.nn as nn

class L2Normalize(nn.Module):
    def __init__(self, net):
        super(L2Normalize, self).__init__()
        self.net = net

    def forward(self, data):
        data = self.net(data)
        norms = data.norm(2, 1)
        batch_size = data.size()[0]
        norms = norms.view(-1, 1).repeat(1, data.size()[1])
        x = data / norms 
        return x

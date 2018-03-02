import torch
import torch.nn as nn

class L2Normalize(nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()

    def forward(self, data):
        norms = data.norm(2, 1)
        #print norms.size()
        batch_size = data.size()[0]
        norms = norms.view(-1, 1).repeat(1, data.size()[1])
        #print norms
        # print (norms.size())
        x = data / norms 
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

def activateFunc(x):
    x = torch.tanh(x)
    return F.relu(x)

class Router(nn.Module):
    def __init__(self, num_out_path, embed_size, hid):
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.mlp = nn.Sequential(nn.Linear(embed_size, hid), 
                                    nn.ReLU(True), 
                                    nn.Linear(hid, num_out_path))
        self.init_weights()

    def init_weights(self):
        self.mlp[2].bias.data.fill_(1.5)

    def forward(self, x):
        x = x.mean(-2)
        x = self.mlp(x)
        soft_g = activateFunc(x) 
        return soft_g

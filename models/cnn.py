import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU()
    )

class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(in_c, 32, kernel_size=3, padding=1, stride = 1),
            conv_block(32, 64, kernel_size=3, padding=1, stride = 1),
            conv_block(64, 128, kernel_size = 3, padding = 1, stride = 2)
        )


        self.decoder = nn.Sequential(
        #Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
            nn.Linear(128 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_classes)
        )


    def forward(self, x):
        x = self.encoder(x)

        x = x.view(x.size(0), -1) # flat

        x = self.decoder(x)
        return x

model = MyCNNClassifier(1, 10)

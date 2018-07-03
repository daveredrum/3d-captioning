import math
import torch
import torch.nn as nn

###################################################################
#                                                                 #
#                                                                 #
#                    model zoo for encoder                        #
#                                                                 #
#                                                                 #
###################################################################

class EmbeddingEncoder(nn.Module):
    def __init__(self):
        super(EmbeddingEncoder, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
    
    def forward(self, inputs):
        outputs = self.fc_layer(inputs)

        return outputs


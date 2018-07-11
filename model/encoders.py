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
            nn.Linear(128, 512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
    
    def forward(self, inputs):
        outputs = self.fc_layer(inputs)

        return outputs

# for attention
class AttentionEncoder(nn.Module):
    def __init__(self):
        super(AttentionEncoder, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=4, stride=4)
        self.global_mapping = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
        )
        self.global_bn = nn.BatchNorm1d(512, momentum=0.01)
        self.area_mapping = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
        )
        self.area_bn = nn.BatchNorm3d(512, momentum=0.01)

    def forward(self, inputs):
        '''
        original_features: (batch_size, 256, 4, 4, 4)
        global_features: (batch_size, 512)
        area_features: (batch_size, 512, 64)
        '''
        batch_size = inputs.size(0)
        original_features = inputs.clone()
        # (batch_size, 512, 196)
        area_features = original_features.permute(0, 2, 3, 4, 1).contiguous()
        area_features = self.area_mapping(area_features).permute(0, 4, 1, 2, 3).contiguous()
        area_features = self.area_bn(area_features).view(batch_size, 512, -1)
        # (batch_size, 512)
        global_features = self.avg_pool(original_features).squeeze()
        global_features = self.global_mapping(global_features)
        global_features = self.global_bn(global_features)

        return original_features, global_features, area_features
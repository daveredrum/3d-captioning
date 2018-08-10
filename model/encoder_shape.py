import torch
import torch.nn as nn

class ShapenetShapeEncoder(nn.Module):
    def __init__(self):
        super(ShapenetShapeEncoder, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv3d(4, 32, 3, stride=1, padding=1), 
        #     nn.ReLU(),
        #     nn.AvgPool3d(2),
        #     nn.BatchNorm3d(32),
        #     nn.Conv3d(32, 64, 3, stride=1, padding=1), 
        #     nn.ReLU(),
        #     nn.AvgPool3d(2),
        #     nn.BatchNorm3d(64),
        #     nn.Conv3d(64, 128, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv3d(128, 128, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm3d(128)
        # )
        self.conv = nn.Sequential(
            nn.Conv3d(4, 64, 3, stride=2, padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(256)
        )
        self.outputs = nn.Linear(256, 128)

    def forward(self, inputs):
        conved = self.conv(inputs).view(inputs.size(0), 256, -1).contiguous().mean(2)
        outputs = self.outputs(conved.view(conved.size(0), -1))

        return outputs

class ShapenetEmbeddingEncoder(nn.Module):
    def __init__(self, encoder_path):
        super(ShapenetEmbeddingEncoder, self).__init__()
        encoder = torch.load(encoder_path)
        self.conv = nn.Sequential(*list(encoder.conv.children())[:-2])
        self.pool = nn.AvgPool3d(2)
        self.bn = encoder.conv[-1]
        self.outputs = encoder.outputs

    def forward(self, inputs):
        area_feat = self.conv(inputs)
        global_feat = self.pool(area_feat)
        global_feat = self.bn(global_feat)
        global_feat = self.outputs(global_feat.view(global_feat.size(0), -1))

        return area_feat, global_feat
        
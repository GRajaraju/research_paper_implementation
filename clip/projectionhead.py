import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.input_size = features.shape[-1]
        self.output_size = 256
        self.linear_layer = nn.Linear(self.input_size, self.output_size)

    def forward(self, embeddings):
        outputs = self.linear_layer(embeddings)
        return outputs
    
    

"""
CLIP is a multi modal (vision and langauge) model that finds the
similarity between an image and its corresponding text description.

Two models are used within CLIP, one that computes the image embeddings
and the other that computes for the text.

Embedding size for the image and the text is different, so we are using a
projection head that resizes the embeddings to same size.
"""

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
    
    

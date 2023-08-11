# Data class
from dataclasses import dataclass

@dataclass
class Config:
    image_embeddings_model: str = "microsoft/resnet-50"
    text_embeddings_model: str = "all-MiniLM-L6-v2"
    epochs: int = 2
    learning_rate: float = 0.001

    
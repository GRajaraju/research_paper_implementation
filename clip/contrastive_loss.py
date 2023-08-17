import torch
import torch.nn as nn
from config import Config


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, image_embeddings, text_embeddings, label):
        margin = Config.contrastive_loss_margin

        distance = torch.sqrt(image_embeddings - text_embeddings)
        distance = torch.mean(torch.nan_to_num(distance))

        loss = (label*distance)**2 + (1 - label) * max((margin - distance), 0) ** 2

        return loss
    
if __name__ == "__main__":

    contrastive_loss = ContrastiveLoss()

    img_input = torch.tensor([1])
    txt_input = torch.tensor([2])

    embeddings = nn.Embedding(100, 512)

    img_emb = embeddings(img_input)
    txt_emb = embeddings(txt_input)

    label = 1

    loss = contrastive_loss(img_emb, txt_emb, label)
    print(loss)
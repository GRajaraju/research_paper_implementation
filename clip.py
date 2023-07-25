from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import BertConfig, BertModel
from PIL import Image

model_name = "microsoft/resnet-50"

processor = AutoImageProcessor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name)



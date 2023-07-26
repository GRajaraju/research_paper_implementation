from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import BertConfig, BertModel
from PIL import Image

model_name = "microsoft/resnet-50"

class ImageEncoder:
    def __init__(self, model_name):
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetForImageClassification.from_pretrained(model_name)

    def features(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(image, return_tensors='pt')
        outputs = self.model(**inputs)

        return outputs.logits

if __name__ == "__main__":

    model_name = "microsoft/resnet-50"
    image_path = "image.png"

    image_encoder = ImageEncoder(model_name)
    image_features = image_encoder.features(image_path)
    print(image_features.shape)


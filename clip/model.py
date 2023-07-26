from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import BertTokenizer, BertModel
from PIL import Image


class ImageEncoder:
    def __init__(self, model_name):
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetForImageClassification.from_pretrained(model_name)

    def features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors='pt')
        outputs = self.model(**inputs)

        return outputs.logits
    
class TextEncoder:
  def __init__(self, model_name):

    self.tokenizer = BertTokenizer.from_pretrained(model_name)
    self.model = BertModel.from_pretrained(model_name)

  def features(self, text):
    inputs = self.tokenizer(text, return_tensors="pt")
    output = self.model(**inputs)
    return output.last_hidden_state


if __name__ == "__main__":

    model_name = "microsoft/resnet-50"
    image_path = "image.png"

    image_encoder = ImageEncoder(model_name)
    image_features = image_encoder.features(image_path)
    print(image_features.shape)

    text_model = "bert-base-uncased"
    text_encoder = TextEncoder(text_model)

    text = "a puppy playing with a football on a green grass"
    outputs = text_encoder.features(text)
    print(outputs.shape)


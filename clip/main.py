import argparse
import json
from model import ImageEncoder, TextEncoder
from projectionhead import ProjectionHead


class TextImageSimilarity:

    def __init__(self, image_model_name, text_model_name):
        
        self.image_encoder = ImageEncoder(image_model_name)
        self.text_encoder = TextEncoder(text_model_name)

    def compute_embeddings(self, image, text):
        if image and text:
            image_features = self.image_encoder.features(image)
            text_features = self.text_encoder.features(text)
        else:
            raise("either image or text is missing")
        
        return image_features, text_features


def main(config_path, image, text):
    
    with open(config_path, 'r') as fr:
        data = json.load(fr)

    text_model = data['text_embeddings']
    image_model = data['image_embeddings']

    similarity = TextImageSimilarity(text_model, image_model)

    score = similarity.compute_embeddings(image, text)
    return score
    

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    parser.add("--config_file_path", help="path of the config file")

    args = parser.parse_args()
    config_path = args.config_file_path

    image_path = "image.png"
    image_description = "a puppy playing with a ball in a green grass"

    score = main(config_path, image_path, image_description)
    
    # model_name = "microsoft/resnet-50"
    # image_encoder = ImageEncoder(model_name)

    # image_features = image_encoder.features(image_path)

    # text_model = "all-MiniLM-L6-v2"
    # text_encoder = TextEncoder(text_model)

    # text_features = text_encoder.features(image_description)

    # image_projection = ProjectionHead(image_features)
    # text_projection = ProjectionHead(text_features)

    # image_embeddings = image_projection.embeddings(image_features)
    # text_embeddings = text_projection.embeddings(text_features)
    


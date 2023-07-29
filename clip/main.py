from model import ImageEncoder, TextEncoder
from projectionhead import ProjectionHead

if __name__ == "__main__":
    
    image_path = "image.png"
    image_description = "a puppy playing with a ball in a green grass"

    model_name = "microsoft/resnet-50"
    image_encoder = ImageEncoder(model_name)

    image_features = image_encoder.features(image_path)

    text_model = "all-MiniLM-L6-v2"
    text_encoder = TextEncoder(text_model)

    text_features = text_encoder.features(image_description)

    image_projection = ProjectionHead(image_features)
    text_projection = ProjectionHead(text_features)

    image_embeddings = image_projection.embeddings(image_features)
    text_embeddings = text_projection.embeddings(text_features)



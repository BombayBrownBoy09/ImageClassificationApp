from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import click

@click.command()
@click.option(
    "--url",
    type=str,
    default="http://images.cocodataset.org/val2017/000000039769.jpg",
)
def classified_image(url):
  url = url
  image = Image.open(requests.get(url, stream=True).raw)

  feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
  model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

  inputs = feature_extractor(images=image, return_tensors="pt")
  outputs = model(**inputs)
  logits = outputs.logits
  # model predicts one of the 1000 ImageNet classes
  predicted_class_idx = logits.argmax(-1).item()
  result = model.config.id2label[predicted_class_idx]
  print("Predicted class:", result)
  return result


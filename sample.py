from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests


font = ImageFont.truetype('Arial.ttf', 20)
def draw(img, coord, label):    
    """
    args:
    image: image
    coord: (left upper x,left upper y,right bottom x,right bottom y)    
    """

    color = (0,255,17)
    
    draw = ImageDraw.Draw(img)
    draw.rectangle(coord, fill=None, outline=color, width=2)
    lu_x, lu_y, _, _ = coord 
    coord = (lu_x - 15, lu_y - 30)
    draw.text(coord, label, color, font=font)
    return img


# image = Image.open("./apple.jpeg")
image = Image.open("./images/ninjin.jpeg")

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    # let's only keep detections with score > 0.9
    if score > 0.9:
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
        img_label = "{}, {}".format(model.config.id2label[label.item()], round(score.item(), 3))
        img = draw(image, tuple(box), img_label)

if img is None:
    print("no detective")
else:
    img.show()


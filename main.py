from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import cv2
import numpy as np

font = ImageFont.truetype('Arial.ttf', 20)
def draw(img, coord, label):    
    """
    args:
    image: image
    coord: (left upper x,left upper y,right bottom x,right bottom y)    
    """

    # color = (0,255,17) # green
    color = (255,0,0) # red
    
    draw = ImageDraw.Draw(img)
    draw.rectangle(coord, fill=None, outline=color, width=2)
    lu_x, lu_y, _, _ = coord 
    coord = (lu_x - 15, lu_y - 30)
    draw.text(coord, label, color, font=font)
    return img


class DetectInfo():
    def __init__(self, label: str, box: tuple[int], score: float):
        self.label = label
        self.box = box
        self.score = score
        self.size = -1
        
    def set_size(self, size: int):
        self.size = size
        
        
def detect(image, target_label, th = 0.9) -> list[DetectInfo]:
    detect_infos = []

    
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
            obj_label = model.config.id2label[label.item()]          
            if obj_label == "carrot":
               d = DetectInfo(obj_label, tuple(box), round(score.item(), 3))
               detect_infos.append(d)

    return detect_infos

def pil2cv(image: Image) :
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def cal_size(img: Image):
    img = pil2cv(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    height, width = img.shape

    ret, binary = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)# 閾値で2値化
    # new_img = cv2pil(binary)
    # new_img.show()

    obj_size = np.count_nonzero(binary == 255)
    return obj_size


def main():
    # image = Image.open("./images/ninjin-rotate.jpeg")
    image = Image.open("./images/ninjin_crop.jpeg")

    
    detect_infos = detect(image, "ca")
    for info in detect_infos:
        cropped = image.crop(info.box)
        # cropped.show()
        size = cal_size(cropped)
        info.set_size(size)

        img_label = "{}, size: {}".format(info.label, info.size)
        img = draw(image, info.box, img_label)
    if img is None:
        print("no detective")
    else:
        img.show()

    
if __name__ == "__main__":
    main()

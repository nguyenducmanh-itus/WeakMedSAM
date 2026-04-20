#This file is designed to visiualize same disease in different organs
import pandas as pd
from PIL import Image, ImageDraw
import math
import os
import json
df = pd.read_excel("./data/BTXRD/dataset.xlsx", sheet_name = "Sheet1")
benign_specific = df["osteochondroma"]
subset = df.loc[:, "hand" : "shoulder-joint"]
image_paths = []
images = []
#Lấy các file .json của ảnh được lấy để trực quan hóa
get_json = []
#Lấy tất cả tổn thương lành tính osteochondroma trên các cơ quan xương khác nhau
for col in subset.columns : 
    filtered = df[(subset[col] == 1) & (benign_specific == 1)]
    if not filtered.empty :
        filtered = filtered.iloc[0]
        get_json.append(filtered["image_id"].removesuffix(".jpeg") + ".json")
        image_path = os.path.join("./data/BTXRD/images", filtered["image_id"])
        image_paths.append(image_path)
        images.append(Image.open(image_path))
#Bounding box vùng tổn thương dựa vào file json
def get_BBox(inf_bbox, image_size, size = (512, 512)) : 
    json_path = os.path.join("./data/BTXRD/Annotations", inf_bbox)
    with open(json_path, "r", encoding = "utf-8") as f :
        data = json.load(f)
    for shape in data["shapes"] :
        if shape["shape_type"] == "rectangle" : 
            points = shape["points"]
            (x1, y1), (x2, y2) = points
            x_scale = size[0] / image_size[0]
            y_scale = size[1] / image_size[1]
            x_min = min(x1, x2) * x_scale
            x_max = max(x1, x2) * x_scale
            y_min = min(y1, y2) * y_scale
            y_max = max(y1, y2) * y_scale
            break
    return x_min, x_max, y_min, y_max
def draw_BBox(images, json_files, size = (512, 512)):
    boxed_images = []
    for image, inf_bbox in zip(images, json_files):
        original_shape = image.size
        resized_image = image.resize(size).convert("RGB")
        x_min, x_max, y_min, y_max = get_BBox(inf_bbox, original_shape)
        draw = ImageDraw.Draw(resized_image)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="yellow", width=1)
        boxed_images.append(resized_image)

    return boxed_images

def visualize_multiple_image(images, json_files, cols=4, size = (512, 512)):
    boxed_images = draw_BBox(images, json_files, size)
    rows = math.ceil(len(boxed_images) / cols)
    visualize = Image.new("RGB", (size[0] * cols, size[1] * rows))

    for i, image in enumerate(boxed_images):
        x = (i % cols) * size[0]
        y = (i // cols) * size[1]
        visualize.paste(image, (x, y))

    visualize.show()

visualize_multiple_image(images, get_json)


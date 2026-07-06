import json 
import os
import pickle
import numpy as np

bbox = []
json_dir = "data/BTXRD/Annotations"
image_dir = "data/BTXRD/images"
json_path = [os.path.join(json_dir, f) for f in os.listdir(json_dir)]
imgs = [f.split(".")[0] for f in os.listdir(image_dir)]
  
for json_file, img_id in zip(json_path, imgs) : 
    with open(json_file, "r") as f :
        data = json.load(f)
    i = 0
    for shape in data["shapes"] :
        
        label = shape["label"]
        shape_type = shape["shape_type"]
        points = shape["points"]

        if shape_type == "rectangle":
            i+=1
            (x1, y1), (x2, y2) = points
            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1, x2)
            ymax = max(y1, y2)
            box = [xmin, ymin, xmax, ymax]
            bbox.append((img_id, box))            
            break

with open("bbox_map.bin", 'rb') as f :
    pred_bbox = pickle.load(f)

list_pred_bbox = []
for keys, values in pred_bbox.items() :
    list_pred_bbox.append((keys, list(values)))
    
def compute_iou(box1, box2):
    """
    box = [x1, y1, x2, y2]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter

    if union == 0:
        return 0.0

    return inter / union

def evaluate_iou(list1, list2):
    dict2 = {img_id: bbox for img_id, bbox in list2}

    results = {}

    for img_id, bbox1 in list1:
        if img_id in dict2:
            results[img_id] = compute_iou(bbox1, dict2[img_id])

    mean_iou = np.mean(list(results.values())) if results else 0.0

    return mean_iou, results

mean_iou, results = evaluate_iou(list_pred_bbox, bbox)
print(mean_iou)
print(results)
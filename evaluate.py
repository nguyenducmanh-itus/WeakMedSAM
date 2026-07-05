import json 
import os
import pickle

bbox = []
json_dir = "data/BTXRD/Annotations"
json_path = [os.path.join(json_dir, f) for f in os.listdir(json_dir)]

for json_file in json_path : 
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
            bbox.append(box)            
            break

with open("bbox_map.bin", 'rb') as f :
    pred_bbox = pickle.load(f)

list_pred_bbox = []
for keys, values in pred_bbox.items() :
    list_pred_bbox.append(values)
    
print(len(list_pred_bbox))


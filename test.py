import pickle
import cv2 as cv
import json

with open("box_crop/box_map.bin", 'rb') as f :
    file = pickle.load(f)

with open("data/BTXRD/Annotations/IMG000003.json") as f_json :
    json_file = json.load(f_json)
    
    
bounding_box = []
for keys, values in file.items() :
    bounding_box.append((keys, values))

shape = json_file["shapes"]
for i in range(len(shape)) :
    if shape[i]["shape_type"] == "rectangle" :
        (x1_gt, y1_gt) , (x2_gt, y2_gt) = shape[i]["points"] 


x1_pred, y1_pred, x2_pred, y2_pred = bounding_box[2][1]


img = cv.imread("data/BTXRD/images/IMG000003.jpeg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
h, w, _ = img.shape
img_resize = cv.resize(img, (512, 512))
x1_pred = int((512 / w) * x1_pred)
x2_pred = int((512 / w) * x2_pred)
x1_gt = int((512 / w) * x1_gt)
x2_gt = int((512 / w) * x2_gt)

y1_pred = int((512 / h) * y1_pred)
y2_pred = int((512 / h) * y2_pred)
y1_gt = int((512 / h) * y1_gt)
y2_gt = int((512 / h) * y2_gt)
#print(x1_gt, y1_gt, x2_gt, y2_gt)
cv.rectangle(img_resize, (x1_pred, y1_pred), (x2_pred, y2_pred), (0, 255, 255), 2)
cv.rectangle(img_resize, (x1_gt, y1_gt), (x2_gt, y2_gt), (255, 255, 255), 2)

cv.imshow("Crop area", img_resize)
cv.waitKey(0)

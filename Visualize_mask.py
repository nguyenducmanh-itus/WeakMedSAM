import os
from PIL import Image
import json
import numpy as np
import cv2 as cv

with open("./data/BTXRD/Annotations/IMG000001.json", "r") as f : 
    data = json.load(f)

#load image
image_path = data["imagePath"]
img = cv.imread(os.path.join("./data/BTXRD/images", image_path))
h = data["imageHeight"]
w = data["imageWidth"]
mask = np.zeros((h, w), dtype=np.uint8)

for shape in data["shapes"] : 
    if shape["shape_type"] == "polygon" : 
        points = np.array(shape["points"], dtype = np.int32)
        points = points.reshape((-1, 1, 2))
        cv.fillPoly(mask, [points], 255)
        
overlay = img.copy()
overlay[mask == 255] = [0, 0, 255]
alpha = 0.5
result = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
mask = cv.resize(mask, (512, 512), interpolation = cv.INTER_AREA)
result = cv.resize(result, (512, 512), interpolation = cv.INTER_AREA)
# ===== 6. Show =====
cv.imshow("Mask", mask)
cv.imshow("Overlay", result)
cv.waitKey(0)
cv.destroyAllWindows()
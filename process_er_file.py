import pandas as pd
import os 

csv_path = './data/FracAtlas/dataset.csv'
img_path = './data/FracAtlas/images'

df = pd.read_csv(csv_path)
df = df["image_id"]
file_extension  = (".jpg")
file_name_fractured = []
for file in os.listdir(os.path.join(img_path, "Fractured")) :
    if file.endswith(file_extension) :
        file_name_fractured.append(file)
        
file_name_none_fractured = []
for file in os.listdir(img_path, "None_fractured") :
    if file.endswith(file_extension) :
        file_name_none_fractured.append(file)
        
missing_img = [img for img in file_name_none_fractured if img not in df.values]

with open('error_file.txt', 'w') as f : 
    for img in missing_img : 
        f.write('%s\n' %img)

for i in missing_img : 
    file_err_path = os.path.join(img_path,"None_fractured", i)
    os.remove(file_err_path)
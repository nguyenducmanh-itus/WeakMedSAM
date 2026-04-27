import os

#This code used for convert all images to same format
data_path = "./data/BTXRD/images"
dir_list = os.listdir("./data/BTXRD/images")
for image in dir_list :
    old_path = os.path.join(data_path, image)
    if os.path.isfile(old_path) :
        name, ext = os.path.splitext(old_path)
        if ext == ".jpg" :
            new_path = name + ".jpeg"
            os.rename(old_path, new_path)
    

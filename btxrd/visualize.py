import json

json_path = "data/BTXRD/Annotations/IMG000001.json"
with open(json_path, mode="r", encoding="utf-8") as read_file :
    tumor_inf = json.load(read_file)

shape_list = []

for i in range(len(tumor_inf["shapes"])) :
    print(tumor_inf["shapes", i, "points"])
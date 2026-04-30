import torch
import pandas as pd

df = pd.read_excel("./data/BTXRD/dataset.xlsx")
# Chọn các cột feature cần kiểm tra
lesion_cols = df.loc[:, "osteochondroma":"other mt"]
df['max_multiple'] = multi_features = lesion_cols.sum(axis = 1)
list_images = df[df["max_multiple"] == 2]["image_id"].tolist()
print(list_images)
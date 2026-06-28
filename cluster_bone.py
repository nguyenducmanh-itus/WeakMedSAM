import pandas as pd
import numpy as np

#15 features bone will be clustered in 4 cluster 
#following similarity in features vision
#Cluster 1 : femur, tibia, fibula, humerus, ulna, radius
#Cluster 2 : hand, foot
#Cluster 3 : hip bone
#Cluster 4 : shoulder-joint, elbow-joint, wrist-joint, hip-joint, knee-joint, ankle-joint

df = pd.read_excel("data/BTXRD/dataset.xlsx")
features_body = df.loc[:, "hand" : "shoulder-joint"].columns
body_cluster = {
    0 : ["femur", "tibia", "fibula", "humerus", "ulna", "radius"], 
    1 : ["hand", "foot"], 
    2 : ["hip bone"], 
    3 : ["shoulder-joint", "elbow-joint", "wrist-joint", "hip-joint", 
         "knee-joint", "ankle-joint"]
}

new_df = []
y = []
for i in range(len(body_cluster)) :
    has_features = (df[body_cluster[i]].any(axis=1))
    is_tumor = (df["tumor"].any())
    new_df.append((has_features & is_tumor).astype(float))
    y.append(new_df[i].to_numpy())
    
y = np.array(y).T
save_features_body_map = {}

for i in range(len(df)) :
    save_features_body_map[df.loc[i, "image_id"]] = y[i] 

for keys, values in save_features_body_map.items() :
    print(f"Image ID : {keys} | labels : {values}")
    









 
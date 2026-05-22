import pandas as pd 
from pathlib import Path

#Get absolute path of current file
current_file = Path(__file__).resolve().parent
#Get data path of excel file
dataset_path = current_file.parent / "data" / "BTXRD" / "dataset.xlsx"
df = pd.read_excel(dataset_path)
part_feature = df.loc[:, "hand" : "shoulder-joint"]
features_body = part_feature.columns
indices = part_feature[part_feature.sum(axis = 1) >= 2].index
df_filtered = df.drop(index=indices)
part_dict = {}
result = df_filtered[(df_filtered['hand'] == 1) & (df_filtered['tumor'] == 0)]
for part in features_body : 
    result = df_filtered[(df_filtered[part] == 1) & (df_filtered['tumor'] == 1)]
    part_dict[part] = len(result)
    result_non = df_filtered[(df_filtered[part] == 1) & (df_filtered['tumor'] == 0)]
    part_dict['non' + ' ' + part] = len(result_non)

    


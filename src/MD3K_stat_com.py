import json
import os
from collections import defaultdict
import cv2

json_folder_RGB = "./output/MD-3K-eval/GeoWizard"
json_folder_LAP = "./output/MD-3K-eval/GeoWizard_lap"
raw_json_path = "./data/MD-3k/annotations.json"

with open(raw_json_path, 'r') as file:
        data_raw = json.load(file)

depth_anything_v2_models = [
        "depth-anything/Depth-Anything-V2-Small-hf",
        "depth-anything/Depth-Anything-V2-Base-hf",
        "depth-anything/Depth-Anything-V2-Large-hf",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    ]

depth_anything_v1_models = [
    "LiheYoung/depth-anything-small-hf",
    "LiheYoung/depth-anything-base-hf",
    "LiheYoung/depth-anything-large-hf",
]

model_configs = [
    {"name": model_name}
    for model_name in depth_anything_v2_models + depth_anything_v1_models
] + [
    {"name": "ZoeDepth"},
    {"name": "DPT"},
]

# Calculate ML-SRA
index = 0
for model_cfg in model_configs:
    json_name = model_cfg['name'].split('/')[-1]
    json_path = os.path.join(json_folder_RGB, f"{json_name}.json")
    with open(json_path, 'r') as file:
        data_RGB = json.load(file)
        
    json_path = os.path.join(json_folder_LAP, f"{json_name}.json")
    with open(json_path, 'r') as file:
        data_LAP = json.load(file)
        
    results = {"correct": 0, "total": 0}
    totalCorrect = 0
    totalPred = 0    
    for item in data_RGB:
        group_name = item.split('/')[1]
        for est in data_RGB[item]:
            if (data_raw[item][0]['label'] ==2): #choose a certain class 1, 2 or both
                totalPred += 1
                results['total'] += 1
                if index > 5:
                    if (est['Correct'] and data_raw[item][0]['label'] == 1 and data_LAP[item][0]['Correct']) or ((not est['Correct']) and data_raw[item][0]['label'] == 2 and ( data_LAP[item][0]['Correct'])):
                        totalCorrect += 1
                        results['correct'] += 1
                else:
                    if (est['Correct'] and data_raw[item][0]['label'] == 1 and data_LAP[item][0]['Correct']) or (( est['Correct']) and data_raw[item][0]['label'] == 2 and (not data_LAP[item][0]['Correct'])):
                        totalCorrect += 1
                        results['correct'] += 1
    index += 1     
    print(f"{json_name} has {totalCorrect} correct pred out of {totalPred}, ratio: {(totalCorrect/totalPred * 100):.1f}\n")


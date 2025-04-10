import json
import os
from collections import defaultdict
import cv2

json_folder = "./output/MD-3K-eval/GeoWizard_lap"
    

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
    {"name": "DPT"}
]

# Calculate SRA1 of the whole dataset
for model_cfg in model_configs:
    json_name = model_cfg['name'].split('/')[-1]
    json_path = os.path.join(json_folder, f"{json_name}.json")
    with open(json_path, 'r') as file:
        data = json.load(file)
    results = {"correct": 0, "total": 0}
    totalCorrect = 0
    totalPred = 0    
    for item in data:
        group_name = item.split('/')[1]
        for est in data[item]:
            totalPred += 1
            results['total'] += 1
            if est['Correct']:
                totalCorrect += 1
                results['correct'] += 1
    
    print(f"{json_name} has {totalCorrect} correct pred out of {totalPred}, ratio: {(totalCorrect/totalPred * 100):.1f}\n")



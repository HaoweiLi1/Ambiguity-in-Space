import json
import os
from collections import defaultdict
import cv2

json_folder = "./output/lap_nocolor_2_f/GeoWizard"

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



for model_cfg in model_configs[-1]:
    # json_name = model_cfg['name'].split('/')[-1]
    json_name = 'GeoWizard'
    json_path = os.path.join(json_folder, f"{json_name}.json")
    
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    results = {
        "indoor": {"correct": 0, "total": 0},
        "outdoor": {"correct": 0, "total": 0},
        "non_real": {"correct": 0, "total": 0},
        "transparent_reflective": {"correct": 0, "total": 0},
        "adverse_style": {"correct": 0, "total": 0},
        "aerial": {"correct": 0, "total": 0},
        "underwater": {"correct": 0, "total": 0},
        "object": {"correct": 0, "total": 0}
    }
    
    totalCorrect = 0
    totalPred = 0
    
    for item in data:
        group_name = item.split('/')[1]
        for est in data[item]:
            totalPred += 1
            results[group_name]['total'] += 1
            if est['Correct']:
                totalCorrect += 1
                results[group_name]['correct'] += 1

    # Now for LaTeX format
    latex_results = []
    for res in results:
        accuracy = (results[res]['correct'] / results[res]['total'] * 100) if results[res]['total'] > 0 else 0
        latex_results.append(f"{accuracy:.1f}")
    
    latex_result_str = " & ".join(latex_results)  # Combine all results with "&"
    
    # Output in LaTeX-friendly format
    print(f"{json_name} & {latex_result_str} & {(totalCorrect / totalPred * 100):.1f} \\\\")            

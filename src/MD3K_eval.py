import json
import os
from collections import defaultdict
import cv2

json_path = "./data/MD-3K/annotations.json"
depth_res_path = "./output/lap_nocolor_2_f"
raw_path = "./data/MD-3K"
output_folder = "./MD-3K-eval/lap_nocolor_2_f"


os.makedirs(output_folder,exist_ok=True)
with open(json_path, 'r') as file:
    data = json.load(file)
    

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

results = defaultdict(lambda: defaultdict(list))
# Process each image and its annotations
for image_path, annotations in data.items():
    raw_img_path = os.path.join(raw_path, image_path[2:])

    group = image_path.split('/')[1]
    image_name = image_path.split('/')[2].split('.')[0]
    for config in model_configs:
        try:
            raw_img = cv2.imread(raw_img_path)
            est_img_path = os.path.join(depth_res_path, group, config['name'], image_name) + '.png'
            est_img = cv2.imread(est_img_path, -1)
            # print(est_img_path)
            est_img = cv2.resize(est_img, (raw_img.shape[1],raw_img.shape[0]))
        except:
            print(f"Fail to open image {est_img_path}")
            continue
        
        for annotation in annotations:
            point1 = annotation['point1']
            point2 = annotation['point2']
            # print(point1,point2)
            dep_1 = est_img[int(point1[1]),int(point1[0])]
            dep_2 = est_img[int(point2[1]),int(point2[0])]    
            if "Metric" in config['name'] or "Zoe" in config["name"]:
                flag = dep_1 < dep_2
            else:
                flag = dep_1 > dep_2
            flag = bool(flag)
            results[config['name'].split('/')[-1]][image_path].append({
                'point1': point1,
                'point2': point2,
                'depth1': int(dep_1),  # Convert numpy data type to Python native type for JSON serialization
                'depth2': int(dep_2),
                'Correct': flag
            })
            
    print(f"Finish {image_name} evalution")

for item in results:
    output_json_path = os.path.join(output_folder, item)+'.json'
    with open(output_json_path, 'w') as f:
        json.dump(results[item], f, indent=4)
    print(f'Finish model: {item} saving.')


            

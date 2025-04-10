import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
matplotlib.use('Agg')

# Set font properties
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20  # Double the font size

def normalize_mask(mask_path, normalized_size=(100, 100)):
    """Normalize mask to [0,1] x [0,1] space."""
    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        return None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error loading mask: {mask_path}")
        return None
    
    binary_mask = (mask > 128).astype(np.float32)
    normalized_mask = cv2.resize(binary_mask, normalized_size)
    return normalized_mask

def create_distribution_map(annotations_file="annotations_self.json", 
                          mask_dir="./mask",
                          output_dir="./spatial_distribution"):
    """Create spatial distribution map in [0,1] x [0,1] space."""
    
    os.makedirs(output_dir, exist_ok=True)
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    normalized_size = (100, 100)
    accumulated_mask = np.zeros(normalized_size[::-1], dtype=np.float32)
    processed_count = 0
    label_counts = {1: 0, 2: 0, 3: 0}
    
    for image_path, image_annotations in annotations.items():
        if any(ann['label'] in [1, 2] for ann in image_annotations):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(mask_dir, f'{base_name}.png')
            
            for ann in image_annotations:
                label_counts[ann['label']] = label_counts.get(ann['label'], 0) + 1
            
            normalized_mask = normalize_mask(mask_path)
            if normalized_mask is not None:
                accumulated_mask += normalized_mask
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} masks")
    
    probability_map = accumulated_mask / processed_count if processed_count > 0 else accumulated_mask
    
    # Create visualization with new styling
    plt.figure(figsize=(10, 10))
    plt.imshow(probability_map, cmap=matplotlib.colormaps.get_cmap('Spectral_r'),
              extent=[0, 1, 1, 0])  # extent for [0,1] x [0,1] space
    

    plt.colorbar(label='Probability of Ambiguous Region')
    
    plt.axis('off')
    
    # Save visualization
    vis_path = os.path.join(output_dir, 'spatial_distribution_map.png')
    plt.savefig(vis_path, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()
    
    # Save raw data
    data_path = os.path.join(output_dir, 'probability_map.npy')
    np.save(data_path, probability_map)
    
    print(f"\nProcessing complete. Files saved to {output_dir}")

if __name__ == "__main__":
    create_distribution_map()
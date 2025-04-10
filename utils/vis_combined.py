import cv2
import numpy as np
import os

def load_and_resize_image(image_path, target_size=None):
    """
    Load an image and resize it if target_size is provided.
    Handles both regular images and depth maps.
    """
    # Read image
    if image_path.endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            return None
            
        # For depth maps stored as 16-bit PNG
        if img.dtype != np.uint8:
            # Normalize depth map for visualization
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        print(f"Unsupported file format: {image_path}")
        return None

    # Resize if target size is provided
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]))

    return img

def create_concatenated_image(image_id):
    """
    Create a concatenated image from 5 images in a row:
    original RGB, depth, generated RGB, depth1, generated RGB1
    """
    # Define paths
    img_path = f"./image/image_720p/{image_id}.jpg"
    depth_path = f"./depth/Depth-Anything-V2-Large-hf_LAP3_second_layer/{image_id}.png"
    rgb_path = f"./image/imagev2_layer2_ctrl_newprompt/{image_id}.png"
    depth1_path = f"./depth/Depth-Anything-V2-Large-hf_RGB_first_layer/{image_id}.png"
    rgb1_path = f"./image/imagev2_layer1_ctrl_newprompt/{image_id}.png"

    # Load depth image first to get target size
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        print(f"Error: Could not load depth image {depth_path}")
        return None

    target_size = depth_img.shape[:2]  # (height, width)

    # Load and process all images
    images = []
    
    # Load original RGB and resize to match depth image
    img = load_and_resize_image(img_path, target_size)
    if img is not None:
        images.append(img)

    # Load depth image and normalize for visualization
    depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    images.append(depth_vis)

    # Load generated RGB
    rgb = load_and_resize_image(rgb_path, target_size)
    if rgb is not None:
        images.append(rgb)

    # Load depth1 and normalize for visualization
    depth1_img = cv2.imread(depth1_path, cv2.IMREAD_UNCHANGED)
    if depth1_img is not None:
        depth1_vis = cv2.normalize(depth1_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth1_vis = cv2.cvtColor(depth1_vis, cv2.COLOR_GRAY2BGR)
        images.append(depth1_vis)

    # Load generated RGB1
    rgb1 = load_and_resize_image(rgb1_path, target_size)
    if rgb1 is not None:
        images.append(rgb1)

    # Check if we have all images
    if len(images) != 5:
        print(f"Error: Could not load all images for ID {image_id}")
        return None

    # Concatenate images horizontally
    concat_img = np.hstack(images)

    # Create output directory if it doesn't exist
    output_dir = "./vis_results"
    os.makedirs(output_dir, exist_ok=True)

    # Save concatenated image
    output_path = os.path.join(output_dir, f"{image_id}.png")
    cv2.imwrite(output_path, concat_img)
    print(f"Saved concatenated image to {output_path}")

    return concat_img

def get_all_image_ids():
    """
    Get all image IDs by scanning the rgb directory.
    Returns a sorted list of IDs.
    """
    rgb_dir = "./image/imagev2_layer1_ctrl_newprompt"
    if not os.path.exists(rgb_dir):
        print(f"Error: Directory {rgb_dir} does not exist")
        return []

    image_ids = []
    for filename in os.listdir(rgb_dir):
        if filename.endswith('.png'):
            # Extract the image ID by removing the extension
            image_id = os.path.splitext(filename)[0]
            image_ids.append(image_id)

    # Sort image IDs numerically
    return sorted(image_ids, key=lambda x: int(x) if x.isdigit() else x)

def process_all_images():
    """Process all images found in the rgb directory"""
    image_ids = get_all_image_ids()
    total_images = len(image_ids)
    
    if total_images == 0:
        print("No images found to process")
        return

    print(f"Found {total_images} images to process")
    
    for i, image_id in enumerate(image_ids, 1):
        print(f"Processing image {image_id} ({i}/{total_images})")
        concat_img = create_concatenated_image(image_id)
        if concat_img is None:
            print(f"Failed to process image {image_id}")

if __name__ == "__main__":
    process_all_images()
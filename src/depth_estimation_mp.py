import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, ZoeDepthForDepthEstimation, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Process
import cv2  # OpenCV is useful for filtering and saving images

# Define the target resolution for 480p (width, height)
TARGET_RESOLUTION = (640, 480)

def normalize_depth(depth_tensor):
    return (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())

def save_depth_image(depth_array, output_path):
    depth_image = Image.fromarray((depth_array * 255).astype("uint8"))
    depth_image.save(output_path)

def save_colored_depth(depth_array, output_path):
    cmap = plt.get_cmap('Spectral_r')
    colored_depth = cmap(depth_array)[:, :, :3]  # Get RGB values
    plt.imsave(output_path, colored_depth)

def process_and_save_depth(image, processor, model, model_name, image_name, output_folder, output_vis_folder, group_sub):
    # Prepare the image for the model
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        inputs = inputs.to(model.device)
        outputs = model(**inputs) if hasattr(model, "forward") else model(pixel_values=inputs["pixel_values"])

    # Post-process outputs depending on the model type
    if model_name == "ZoeDepth":
        post_processed_output = processor.post_process_depth_estimation(
            outputs,
            source_sizes=[(image.height, image.width)]
        )
    else:
        post_processed_output = processor.post_process_depth_estimation(outputs)

    predicted_depth = post_processed_output[0]["predicted_depth"]
    normalized_depth = normalize_depth(predicted_depth).detach().cpu().numpy()

    print(f"{model_name} - Depth map shape: {normalized_depth.shape}")

    # Convert depth to 16-bit integer
    depth_png_output_path = os.path.join(output_folder, group_sub, model_name, f"{image_name}.png")
    os.makedirs(os.path.dirname(depth_png_output_path), exist_ok=True)
    depth_16bit = (normalized_depth * 65535).astype(np.uint16)

    # Save the 16-bit PNG image
    cv2.imwrite(depth_png_output_path, depth_16bit)

    # Save the colored depth visualization
    vis_output_path = os.path.join(output_vis_folder, group_sub, model_name, f"{image_name}.jpg")
    os.makedirs(os.path.dirname(vis_output_path), exist_ok=True)
    save_colored_depth(normalized_depth, vis_output_path)

def process_single_model(input_folder, config, output_folder, output_vis_folder, device, gau_size, lvp_type):
    """
    Processes images for a single model configuration on a specified GPU.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        config (dict): Model configuration with 'name', 'processor', and 'model'.
        output_folder (str): Path to save depth estimation results.
        output_vis_folder (str): Path to save visualization results.
        device (str): Device to use for computation (e.g., 'cuda:0').
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_vis_folder, exist_ok=True)

    print(f"Initializing processor and model {config['name']} on {device}...")
    processor = AutoImageProcessor.from_pretrained(config["processor"])
    model = config["model"].from_pretrained(config["processor"]).to(device)

    total_images = sum(
        1 for dirpath, _, filenames in os.walk(input_folder)
        for filename in filenames if filename.lower().endswith(('.png', '.jpg', '.jpeg' , '.webp'))
    )

    image_counter = 0
    for dirpath, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')):
                image_counter += 1
                image_path = os.path.join(dirpath, filename)

                with open(image_path, 'rb') as f:
                    image = Image.open(f).convert('RGB')
                    image = image.resize(TARGET_RESOLUTION, Image.ANTIALIAS)
                    img_rgb = np.array(image)
                    
                    if gau_size != 0:
                        smoothed_img = cv2.GaussianBlur(img_rgb, (gau_size, gau_size), 0)
                        smoothed_img_bgr = cv2.cvtColor(smoothed_img, cv2.COLOR_RGB2BGR)
                        smoothed_img_rgb = cv2.cvtColor(smoothed_img_bgr, cv2.COLOR_BGR2RGB)
                        input_image = Image.fromarray(smoothed_img_rgb)
                    
                    if lvp_type == 0:
                        center_value = -4
                        sharpen_kernel = np.array([
                            [0, -center_value/4, 0],
                            [-center_value/4, center_value, -center_value/4],
                            [0, -center_value/4, 0]
                        ], dtype=np.float32)
                        sharpened_img = cv2.filter2D(img_rgb, -1, sharpen_kernel)
                        sharpened_img_bgr = cv2.cvtColor(sharpened_img, cv2.COLOR_RGB2BGR)
                        sharpened_img_rgb = cv2.cvtColor(sharpened_img_bgr, cv2.COLOR_BGR2RGB)
                        input_image = Image.fromarray(sharpened_img_rgb)
                        
                    #LAP-2
                    if lvp_type == 1:
                        center_value = 8
                        sharpen_kernel = np.array([
                            [-center_value/8, -center_value/8, -center_value/8],
                            [-center_value/8, center_value, -center_value/8],
                            [-center_value/8, -center_value/8, -center_value/8]
                        ], dtype=np.float32)
                        sharpened_img = cv2.filter2D(img_rgb, -1, sharpen_kernel)
                        input_image = Image.fromarray(sharpened_img)

                    #LAP-R
                    if lvp_type == 2:
                        center_value = 4
                        sharpen_kernel = np.array([
                            [0, -center_value/4, 0],
                            [-center_value/4, center_value, -center_value/4],
                            [0, -center_value/4, 0]
                        ], dtype=np.float32)
                        sharpened_img = cv2.filter2D(img_rgb, -1, sharpen_kernel)
                        sharpened_img_bgr = cv2.cvtColor(sharpened_img, cv2.COLOR_RGB2BGR)
                        sharpened_img_rgb = cv2.cvtColor(sharpened_img_bgr, cv2.COLOR_BGR2RGB)
                        input_image = Image.fromarray(sharpened_img_rgb)

                    #LAP-G
                    if lvp_type == 3:
                        img_gray = np.array(image.convert('L'))
                        center_value = -4
                        sharpen_kernel = np.array([
                            [0, -center_value/4, 0],
                            [-center_value/4, center_value, -center_value/4],
                            [0, -center_value/4, 0]
                        ], dtype=np.float32)
                        laplacian_img = cv2.filter2D(img_gray, cv2.CV_64F, sharpen_kernel)
                        laplacian_img_abs = cv2.convertScaleAbs(laplacian_img)
                        laplacian_img_3channel = np.stack([laplacian_img_abs] * 3, axis=-1)
                        input_image = Image.fromarray(laplacian_img_3channel)

                    image_name = os.path.splitext(filename)[0]
                    group_sub = os.path.basename(dirpath)

                    print(f"[{config['name']}] Processing image {image_counter}/{total_images} on {device}: {image_path}...")
                    process_and_save_depth(
                        input_image, processor, model, config["name"], image_name,
                        output_folder, output_vis_folder, group_sub
                    )
                    print(f"[{config['name']}] Finished image {image_counter}/{total_images} on {device}: {image_name}")

    print(f"[{config['name']}] Finished processing on {device}. Results saved in {output_folder} and visualizations in {output_vis_folder}")

def main():
    # Input folder setup
    input_folder = "./data/DA-2K/images"
    output_folder = "./output/DA2K_lap_results"
    output_vis_folder = "./output/DA2K_lap_vis_results"

    # Model configurations
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
        {"name": model_name, "processor": model_name, "model": AutoModelForDepthEstimation}
        for model_name in depth_anything_v2_models + depth_anything_v1_models
    ] + [
        {"name": "ZoeDepth", "processor": "Intel/zoedepth-nyu-kitti", "model": ZoeDepthForDepthEstimation},
        {"name": "DPT", "processor": "Intel/dpt-large", "model": DPTForDepthEstimation},
    ]

    # Configure GPUs
    num_gpus = 1
    gpu_devices = [f"cuda:{i}" for i in range(num_gpus)]

    # Spawn one process per model
    processes = []
    gau_size = 0
    lvp_type = -1
    for idx, config in enumerate(model_configs):
        gpu = gpu_devices[idx % num_gpus]  # Cycle through available GPUs
        process = Process(
            target=process_single_model,
            args=(input_folder, config, output_folder, output_vis_folder, gpu, gau_size, lvp_type)
        )
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("All model configurations processed.")

if __name__ == "__main__":
    main()


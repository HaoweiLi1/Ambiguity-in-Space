import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2  # OpenCV to load and resize images

import matplotlib
matplotlib.use("Agg")

# Initialize global variables
annotations_data = {}
image_paths = []  # List of image file paths

def load_image(img_path):
    """Load and resize the image to fit the window size (720p) while maintaining aspect ratio."""
    img = cv2.imread(img_path)
    if img is None:
        print("Error loading image")
        return None
    # Resize the image to fit into 1280x720 while maintaining aspect ratio
    height, width = img.shape[:2]
    aspect_ratio = width / height

    if width > height:
        new_width = 1280
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 720
        new_width = int(new_height * aspect_ratio)

    resized_img = cv2.resize(img, (new_width, new_height))

    return resized_img

def create_glass_mask(mask_img):
    """Create a mask image with purple for the glass region (1) and white for the non-mask region (0)."""
    mask_np = np.array(mask_img)  # Convert to numpy array (binary: 0 or 1)
    # Create an image with white background (non-mask region)
    glass_mask = np.ones((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8) * 255  # White background (RGBA)

    # Set mask regions (where mask is 1) to purple (151, 51, 255, 80)
    glass_mask[mask_np == 255] = [151, 51, 255, 80]  # Purple color with transparency

    return Image.fromarray(np.uint8(glass_mask))

def apply_transparency(img, alpha=0.3):
    """Apply transparency to the image (blend RGB with alpha)."""
    img = np.array(img)
    # Apply transparency by blending the RGB values with the given alpha
    img[:, :, :3] = img[:, :, :3] * alpha + (1 - alpha) * 255  # Blend RGB with transparency factor
    return Image.fromarray(img.astype(np.uint8))

def add_black_border(image, border_size=10):
    """Add a black border around an image."""
    # Get the size of the original image
    width, height = image.size
    # Create a new image with a black border
    new_image = Image.new("RGBA", (width + 2 * border_size, height + 2 * border_size), (0, 0, 0, 255))
    # Paste the original image in the center of the new image
    new_image.paste(image, (border_size, border_size))
    return new_image

def draw_labeled_points(ax, point1, point2, color1, color2):
    """Draw labeled points with a cross inside a circle and add a border."""
    # Define point size and border width
    size = 300  # Size of the point (larger)
    edge_width = 3  # Width of the edge (border)
    marker_size = 25  # Size of the marker
    cross_marker = '+'  # Cross marker symbol inside the circle
    
    # Draw the first point as a cross inside a circle with border
    ax.scatter(point1[0], point1[1], marker=cross_marker, color=color1, s=size, edgecolors='black', linewidth=edge_width, label="Point 1")
    
    # Draw the second point as a cross inside a circle with border
    ax.scatter(point2[0], point2[1], marker=cross_marker, color=color2, s=size, edgecolors='black', linewidth=edge_width, label="Point 2")

def visualize_image_and_annotations(image_path, annotations):
    """Visualize the image, mask, and annotations."""
    # Load original image (with resizing)
    img = load_image(image_path)
    if img is None:
        return
    
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for PIL
    
    # Load corresponding mask image (glass region)
    mask_path = os.path.join('./mask', os.path.splitext(os.path.basename(image_path))[0] + '.png')
    mask_img = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else None  # Binary mask (grayscale)

    # Create the mask with purple for the glass region (1) and white for non-mask (0)
    glass_mask_img = create_glass_mask(mask_img) if mask_img else None

    # Add black borders to each image (subfigures)
    img_with_border = add_black_border(img)
    if glass_mask_img:
        glass_mask_with_border = add_black_border(glass_mask_img)
    else:
        glass_mask_with_border = None

    # Create subplots (1x4 grid, all in a single row)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Adjusted for 1 row, 4 columns

    # Subfigure 1: Original RGB image with black border
    axs[0].imshow(img_with_border)
    axs[0].axis('off')

    # Subfigure 2: Glass region (mask with purple) with black border
    if glass_mask_with_border:
        axs[1].imshow(glass_mask_with_border)
        axs[1].axis('off')

    # Subfigure 3: Points in red with transparency and black border
    img_with_transparency = apply_transparency(np.array(img), alpha=0.3)
    img_with_transparency_with_border = add_black_border(img_with_transparency)
    axs[2].imshow(img_with_transparency_with_border)
    axs[2].axis('off')
    for annotation in annotations:
        point1 = annotation["point1"]
        point2 = annotation["point2"]
        draw_labeled_points(axs[2], point1, point2, 'darkred', 'lightcoral')

    # Subfigure 4: Points based on label with transparency and black border
    img_with_transparency = apply_transparency(np.array(img), alpha=0.3)
    img_with_transparency_with_border = add_black_border(img_with_transparency)
    axs[3].imshow(img_with_transparency_with_border)
    axs[3].axis('off')
    for annotation in annotations:
        point1 = annotation["point1"]
        point2 = annotation["point2"]
        label = annotation["label"]

        if label == 1:
            # point1 darker blue, point2 lighter blue
            draw_labeled_points(axs[3], point1, point2, 'darkblue', 'lightblue')
        elif label == 2:
            # point1 lighter blue, point2 darker blue
            draw_labeled_points(axs[3], point1, point2, 'lightblue', 'darkblue')
        elif label == 3:
            # Omit visualization for label == 3
            continue

    # Adjust spacing between subfigures to be very small
    plt.subplots_adjust(wspace=0.05, hspace=0)  # Very small horizontal space

    # Create the 'vis' folder if it doesn't exist
    if not os.path.exists('./vis'):
        os.makedirs('./vis')

    # Save the figure with a smaller padding and transparent background
    save_path = os.path.join('./vis', os.path.basename(image_path).replace('.png', '_visualized.png'))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, transparent=True, dpi=400)

    # Show the plot
    #plt.show()

def process_images(image_folder):
    """Main function to process and visualize all images in the folder."""
    global image_paths
    # Get all images in the folder
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    # Load existing annotations from file if it exists
    annotations_file = "annotations_self.json"  # Path to the annotations JSON file
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            global annotations_data
            annotations_data = json.load(f)

    # Process each image and its annotations
    for image_path in image_paths:
        annotations = annotations_data.get(image_path, [])
        visualize_image_and_annotations(image_path, annotations)

# Example usage
if __name__ == "__main__":
    image_folder = "./image/"  # Your image folder path
    process_images(image_folder)
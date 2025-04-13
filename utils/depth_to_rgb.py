import os
import numpy as np
import cv2
from PIL import Image
import torch
from diffusers import (
    ControlNetModel, 
    StableDiffusionXLControlNetPipeline, 
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler, 
    AutoencoderKL
)
from diffusers.utils import load_image


class DepthChecker:
    """
    A class for analyzing and preprocessing a depth map.
    """
    
    def __init__(self):
        """Initialize the DepthChecker with target parameters."""
        self.target_size = (1088, 896)  # Recommended size
        self.min_size = 512  # Minimum size requirement
        self.max_size = 2048  # Maximum size requirement
    
    def load_image(self, image_path):
        """
        Try to load an image using multiple methods.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (numpy.ndarray, str) - The image array and the loader used
        """
        # Try using PIL first
        try:
            img_pil = Image.open(image_path)
            img_np = np.array(img_pil)
            return img_np, "PIL"
        except Exception:
            pass
        
        # Try using OpenCV
        try:
            img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_cv is not None:
                return img_cv, "OpenCV"
        except Exception:
            pass
            
        raise ValueError(f"Could not load image: {image_path}")

    def analyze_depth_image(self, image_path):
        """
        Analyze the properties of a depth image.
        
        Args:
            image_path (str): Path to the depth image
            
        Returns:
            dict: Analysis results including shape, dtype, value ranges, etc.
        """
        print(f"\nAnalyzing depth image: {os.path.basename(image_path)}")
        print("-" * 50)
        
        # Load image
        img, loader = self.load_image(image_path)
        print(f"Loading method: {loader}")
        
        # Basic information
        print(f"\n1. Basic Information:")
        print(f"Shape: {img.shape}")
        print(f"Data type: {img.dtype}")
        
        # Check size requirements
        h, w = img.shape[:2]
        size_check = (h >= self.min_size and w >= self.min_size and 
                     h <= self.max_size and w <= self.max_size)
        print(f"\n2. Size Check:")
        print(f"Current size: {w}x{h}")
        print(f"Within allowed range ({self.min_size}-{self.max_size}): {size_check}")
        print(f"Recommended size: {self.target_size[0]}x{self.target_size[1]}")
        
        # Value range analysis
        print(f"\n3. Value Range Analysis:")
        min_val = np.min(img)
        max_val = np.max(img)
        mean_val = np.mean(img)
        print(f"Minimum value: {min_val}")
        print(f"Maximum value: {max_val}")
        print(f"Mean value: {mean_val:.2f}")
        
        # Channel check
        print(f"\n4. Channel Check:")
        if len(img.shape) == 2:
            print("Single-channel grayscale image ✓")
            channel_check = True
        elif len(img.shape) == 3 and img.shape[2] == 1:
            print("Single-channel image (3D array) ✓")
            channel_check = True
        else:
            print("Warning: Image is not single-channel")
            channel_check = False
        
        # Summary
        print(f"\n5. Compatibility Summary:")
        if channel_check and size_check:
            print("✓ Image format is generally compatible")
            if img.dtype != np.uint8:
                print("- Recommendation: Normalize value range")
            if h != self.target_size[1] or w != self.target_size[0]:
                print("- Recommendation: Resize to recommended dimensions")
        else:
            print("✗ Image needs conversion to be usable")
            
        # Return analysis results as a dictionary
        return {
            "size_check": size_check,
            "channel_check": channel_check,
            "min_val": min_val,
            "max_val": max_val,
            "mean_val": mean_val,
            "shape": img.shape,
            "dtype": str(img.dtype)
        }

    def preprocess_depth_image(self, image_path, output_path=None):
        """
        Preprocess a depth image to meet model requirements.
        
        Args:
            image_path (str): Path to the depth image
            output_path (str, optional): Path to save the preprocessed image
            
        Returns:
            numpy.ndarray: Preprocessed depth image
        """
        img, _ = self.load_image(image_path)
        
        # Ensure single channel
        if len(img.shape) == 3:
            img = img[:,:,0]
            
        # Normalize to 0-255 range
        if img.dtype != np.uint8:
            img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
            
        # Resize to target size
        img_resized = cv2.resize(img, self.target_size)
        
        # Save if output path is provided
        if output_path:
            cv2.imwrite(output_path, img_resized)
            print(f"Preprocessed image saved to: {output_path}")
            
        return img_resized


class ZoeDepthToRGB:
    """
    Converts a depth map to RGB image using Stable Diffusion XL with ControlNet for Zoe-depth.
    """
    
    def __init__(self):
        """Initialize the converter with model settings."""
        # Initialize depth checker
        self.depth_checker = DepthChecker()
        
        # Initialize model
        self.setup_model()
        
    def setup_model(self):
        """Setup SDXL-ControlNet model with appropriate settings."""
        try:
            # Load ControlNet
            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-zoe-depth-sdxl-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            # Load VAE
            self.vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", 
                torch_dtype=torch.float16
            )
            
            # Setup pipeline
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=self.controlnet,
                vae=self.vae,
                variant="fp16",
                use_safetensors=True,
                torch_dtype=torch.float16
            )
            
            # Enable model CPU offload to save VRAM
            self.pipe.enable_model_cpu_offload()
            
            self.model_loaded = True
            print("Zoe model initialized successfully")
            
        except Exception as e:
            self.model_loaded = False
            print(f"Error initializing Zoe model: {str(e)}")
    
    def generate_image(self, depth_image_path, output_path, prompt=None, seed=None):
        """
        Generate RGB image from depth image with specified parameters.
        
        Args:
            depth_image_path (str): Path to the input depth image
            output_path (str): Path to save the output image
            prompt (str, optional): Text prompt for generation
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            PIL.Image or None: The generated image, or None if generation failed
        """
        if not self.model_loaded:
            print("Model is not properly initialized.")
            return None
            
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
            # Check and preprocess depth image if needed
            analysis = self.depth_checker.analyze_depth_image(depth_image_path)
            if not analysis["size_check"] or not analysis["channel_check"]:
                print("Preprocessing depth image to meet requirements...")
                temp_path = os.path.join(os.path.dirname(output_path), f"temp_{os.path.basename(depth_image_path)}")
                self.depth_checker.preprocess_depth_image(depth_image_path, temp_path)
                depth_image = load_image(temp_path)
            else:
                # Load depth image directly
                depth_image = load_image(depth_image_path)
            
            # Set default prompt if none provided
            if prompt is None:
                prompt = ("a bright, well-lit photograph of an interior space with natural daylight, "
                         "clear windows, balanced lighting, accurate geometry and structure, photorealistic, "
                         "vibrant colors, modern interior design, clean and airy space")
            
            # Set negative prompt
            negative_prompt = (
                "blur, hazy, deformed, disfigured, bad architecture, deformed structure, "
                "poor geometry, text, watermark, distorted, unrealistic lighting"
            )
            
            # Set up generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator("cuda").manual_seed(seed)
            
            # Set generation parameters
            params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": depth_image,
                "num_inference_steps": 50,
                "controlnet_conditioning_scale": 0.8,
                "generator": generator,
                "guidance_scale": 8.0,
            }
            
            # Generate image
            image = self.pipe(**params).images[0]
            
            # Save result
            image.save(output_path)
            print(f"Generated image saved to: {output_path}")
            
            return image
            
        except Exception as e:
            print(f"Error generating image from {depth_image_path}: {str(e)}")
            return None
    
    # ADD BATCH PROCESSING METHOD
    def process_directory(self, input_dir, output_dir, prompt=None, seed=None, file_extension=".png"):
        """
        Process all depth images in a directory.
        
        Args:
            input_dir (str): Directory containing depth images
            output_dir (str): Directory to save output images
            prompt (str, optional): Text prompt for generation
            seed (int, optional): Random seed for reproducibility
            file_extension (str, optional): File extension to filter images
            
        Returns:
            int: Number of processed images
        """
        if not self.model_loaded:
            print("Model is not properly initialized.")
            return 0
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all depth files
        depth_files = [f for f in os.listdir(input_dir) 
                      if os.path.isfile(os.path.join(input_dir, f)) and 
                      f.lower().endswith(file_extension.lower())]
        
        total_files = len(depth_files)
        print(f"Found {total_files} depth images to process in {input_dir}")
        
        # Process each file
        processed_count = 0
        
        for i, filename in enumerate(sorted(depth_files)):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_rgb{file_extension}")
            
            # Skip if output already exists
            if os.path.exists(output_path):
                print(f"Skipping {filename} - output already exists at {output_path}")
                continue
            
            print(f"\nProcessing [{i+1}/{total_files}]: {filename}")
            
            # Generate image
            if self.generate_image(input_path, output_path, prompt, seed) is not None:
                processed_count += 1
                print(f"Progress: {processed_count}/{total_files}")
            
            # Clear CUDA cache periodically
            if hasattr(torch, 'cuda') and i % 5 == 0 and i > 0:
                torch.cuda.empty_cache()
        
        print("\nProcessing complete!")
        print(f"Successfully processed: {processed_count}/{total_files}")
        
        return processed_count


class ControlNetDepthToRGB:
    """
    Implementation of depth-to-RGB conversion using ControlNet with SD 1.5.
    """
    
    def __init__(self):
        """
        Initialize the ControlNet depth-to-RGB converter.
        """
        # Initialize depth checker
        self.depth_checker = DepthChecker()
        
        # Initialize model
        self.setup_model()
    
    def setup_model(self):
        """Setup the ControlNet model."""
        try:
            # Path to model checkpoints
            base_model_path = "runwayml/stable-diffusion-v1-5"
            controlnet_path = "lllyasviel/sd-controlnet-depth"  # Using the standard depth controlnet
            
            # Load ControlNet
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_path, 
                torch_dtype=torch.float16
            )
            
            # Setup pipeline - IMPORTANT: Use StableDiffusionControlNetPipeline for SD 1.5
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(  # Changed pipeline class
                base_model_path, 
                controlnet=self.controlnet, 
                torch_dtype=torch.float16
            )
            
            # Set up scheduler for faster processing
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            # Enable model CPU offload
            self.pipe.enable_model_cpu_offload()
            
            self.model_loaded = True
            print("ControlNet model initialized successfully")
            
        except Exception as e:
            self.model_loaded = False
            print(f"Error initializing ControlNet model: {str(e)}")
    
    def generate_image(self, depth_image_path, output_path, prompt=None, seed=None):
        """
        Generate RGB image from depth image with ControlNet.
        
        Args:
            depth_image_path (str): Path to the input depth image
            output_path (str): Path to save the output image
            prompt (str, optional): Text prompt for generation
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            PIL.Image or None: The generated image, or None if generation failed
        """
        if not self.model_loaded:
            print("Model is not properly initialized.")
            return None
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Check and preprocess depth image if needed
            analysis = self.depth_checker.analyze_depth_image(depth_image_path)
            if not analysis["size_check"] or not analysis["channel_check"]:
                print("Preprocessing depth image...")
                temp_path = os.path.join(os.path.dirname(output_path), f"temp_{os.path.basename(depth_image_path)}")
                self.depth_checker.preprocess_depth_image(depth_image_path, temp_path)
                depth_image = load_image(temp_path)
            else:
                depth_image = load_image(depth_image_path)
            
            # Set default prompt if none provided
            if prompt is None:
                prompt = ("a bright, well-lit photograph of an interior space with natural daylight, "
                        "clear windows, balanced lighting, accurate geometry and structure, photorealistic")
            
            # Set negative prompt
            negative_prompt = "blur, deformed, disfigured, bad architecture, poor geometry, text, watermark"
            
            # Set generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator("cuda").manual_seed(seed)
            
            # Generate image
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                image=depth_image,
                num_inference_steps=50,
                generator=generator,
                guidance_scale=8.0,
                controlnet_conditioning_scale=0.8
            ).images[0]
            
            # Save image
            image.save(output_path)
            print(f"Generated image saved to: {output_path}")
            
            return image
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    # ADD BATCH PROCESSING METHOD
    def process_directory(self, input_dir, output_dir, prompt=None, seed=None, file_extension=".png"):
        """
        Process all depth images in a directory with ControlNet.
        
        Args:
            input_dir (str): Directory containing depth images
            output_dir (str): Directory to save output images
            prompt (str, optional): Text prompt for generation
            seed (int, optional): Random seed for reproducibility
            file_extension (str, optional): File extension to filter images
            
        Returns:
            int: Number of processed images
        """
        if not self.model_loaded:
            print("Model is not properly initialized.")
            return 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all depth files
        depth_files = [f for f in os.listdir(input_dir) 
                      if os.path.isfile(os.path.join(input_dir, f)) and 
                      f.lower().endswith(file_extension.lower())]
        
        total_files = len(depth_files)
        print(f"Found {total_files} depth images to process in {input_dir}")
        
        # Process each file
        processed_count = 0
        
        for i, filename in enumerate(sorted(depth_files)):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_rgb{file_extension}")
            
            # Skip if output already exists
            if os.path.exists(output_path):
                print(f"Skipping {filename} - output already exists at {output_path}")
                continue
            
            print(f"\nProcessing [{i+1}/{total_files}]: {filename}")
            
            # Generate image
            if self.generate_image(input_path, output_path, prompt, seed) is not None:
                processed_count += 1
                print(f"Progress: {processed_count}/{total_files}")
            
            # Clear CUDA cache periodically
            if hasattr(torch, 'cuda') and i % 5 == 0 and i > 0:
                torch.cuda.empty_cache()
        
        print("\nProcessing complete!")
        print(f"Successfully processed: {processed_count}/{total_files}")
        
        return processed_count


def main():
    """
    Main function to convert depth images to RGB using either Zoe or ControlNet model.
    Can process single images or entire directories.
    """
    # Configuration
    model_type = "controlnet"  # Choose "zoe" or "controlnet"
    process_mode = "batch"  # Choose "single" or "batch"
    
    # Paths for single image processing
    depth_image_path = "/home/haowei/Documents/Ambiguity-in-Space/utils/depth/1.png"
    output_path = "./output_rgb.png"
    
    # Paths for batch processing
    input_dir = "/home/haowei/Documents/Ambiguity-in-Space/utils/depth"
    output_dir = "./output_rgb_batch_controlnet"
    
    # Generation parameters
    prompt = "a bright, well-lit photograph of an interior space with natural daylight, clear windows, balanced lighting, accurate geometry and structure, photorealistic"
    seed = 42
    
    print(f"Processing with {model_type.upper()} model in {process_mode.upper()} mode")
    
    # Initialize the appropriate model
    if model_type.lower() == "zoe":
        converter = ZoeDepthToRGB()
    elif model_type.lower() == "controlnet":
        converter = ControlNetDepthToRGB()
    else:
        print(f"Unknown model type: {model_type}. Please use 'zoe' or 'controlnet'")
        return
    
    # Process according to the selected mode
    if process_mode.lower() == "single":
        converter.generate_image(depth_image_path, output_path, prompt, seed)
    elif process_mode.lower() == "batch":
        converter.process_directory(input_dir, output_dir, prompt, seed)
    else:
        print(f"Unknown process mode: {process_mode}. Please use 'single' or 'batch'")
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
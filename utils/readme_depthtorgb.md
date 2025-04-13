# DepthToRGB

A Python utility for converting depth maps to realistic RGB images using AI-powered models.

## Overview

This tool provides two different approaches for depth-to-RGB conversion:

1. **Zoe-based approach**: Uses Stable Diffusion XL with a ControlNet model trained on Zoe-depth maps.
2. **ControlNet-based approach**: Uses Stable Diffusion 1.5 with a standard depth ControlNet model.

The script supports both single image processing and batch processing of entire directories.

## Dependencies

```
pip install torch diffusers transformers opencv-python numpy pillow safetensors accelerate
```


## Usage

### Single Image Processing

```python
python depth_to_rgb.py
```

The script is configured in the `main()` function. By default, it will:
- Process a single depth map image
- Use either Zoe or ControlNet model (configurable)
- Save the output as `output_rgb.png`

### Batch Processing

Set `process_mode` to `"batch"` in the `main()` function to process entire directories:

```python
# In main():
process_mode = "batch"
input_dir = "/path/to/depth/maps"
output_dir = "/path/to/output/folder"
```

### Configuration Options

Edit these variables in the `main()` function:

- `model_type`: Choose `"zoe"` or `"controlnet"`
- `process_mode`: Choose `"single"` or `"batch"`
- `depth_image_path`: Path to input depth map for single processing
- `output_path`: Where to save output for single processing
- `input_dir`: Directory with depth maps for batch processing
- `output_dir`: Where to save batch processing outputs
- `prompt`: Text prompt to guide image generation
- `seed`: Random seed for reproducible results

## Features

- **Depth Image Analysis**: Checks compatibility and suggests preprocessing
- **Automatic Preprocessing**: Normalizes and resizes depth maps as needed
- **Memory Efficient**: Uses model offloading to reduce VRAM requirements
- **Error Handling**: Gracefully handles errors during processing
- **Batch Processing**: Efficiently processes entire directories with progress tracking

## Models

The script will automatically download the required models when first run:

- Zoe approach: `diffusers/controlnet-zoe-depth-sdxl-1.0` + `stable-diffusion-xl-base-1.0`
- ControlNet approach: `lllyasviel/sd-controlnet-depth` + `runwayml/stable-diffusion-v1-5`


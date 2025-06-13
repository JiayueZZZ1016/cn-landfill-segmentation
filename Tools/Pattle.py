import os
import rasterio
import numpy as np
from rasterio.enums import Resampling
from PIL import Image
from tqdm import tqdm

def hex_to_rgb(hex_color):
    """Convert a hex color to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def apply_palette(input_file, palette, output_file):
    """Apply a color palette to a single-band TIFF image and save as PNG."""
    # Open the input TIFF file
    with rasterio.open(input_file) as dataset:
        data = dataset.read(1)  # Read the first band as array

    # Convert data to uint8 to match the palette index range
    data = data.astype(np.uint8)

    # Convert hex palette to RGB array
    palette_rgb = np.array([hex_to_rgb(color) for color in palette], dtype=np.uint8)

    # Ensure data values don't exceed the palette length
    max_label = len(palette) - 1
    data = np.clip(data, 0, max_label)

    # Map data to RGB values
    rgb_data = palette_rgb[data]

    # Convert to PIL Image and save as PNG
    rgb_image = Image.fromarray(rgb_data, 'RGB')
    rgb_image.save(output_file)
    # print(f"Saved: {output_file}")

def process_directory(input_dir, output_dir, palette):
    """Process all TIFF files in the input directory and apply color palette."""
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith('.tif'):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.splitext(os.path.join(output_dir, relative_path))[0] + ".png"

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # Apply palette and save
                apply_palette(input_file, palette, output_file)

# 输入目录
input_directory = "验证点/masks"
output_directory = "验证点/masks_vis"

os.makedirs(output_directory, exist_ok=True)

# 定义调色板（使用16进制颜色表示）
palette = [
    "#000000",  # Class 0: background
    "#FFFFFF",  # Class 1: white
]

# 处理目录
process_directory(input_directory, output_directory, palette)
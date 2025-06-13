import os
import rasterio
import numpy as np
from rasterio.windows import Window

def generate_valid_windows(label_tif, crop_size, num_samples=4, max_attempts=1000):
    """生成满足条件的随机窗口坐标列表"""
    with rasterio.open(label_tif) as src:
        label_array = src.read(1)
        H, W = label_array.shape  # 获取图像高度和宽度

    # 检查图像尺寸是否满足裁剪要求
    if H < crop_size or W < crop_size:
        print(f"Warning: Image {label_tif} size ({W}x{H}) smaller than crop size {crop_size}")
        return []

    valid_windows = []
    attempts = 0

    total_pixels = np.sum(label_array == 1)

    while len(valid_windows) < num_samples and attempts < max_attempts:
        # 生成随机左上角坐标
        x = np.random.randint(0, W - crop_size + 1)
        y = np.random.randint(0, H - crop_size + 1)
        
        # 提取标签区域并计算条件
        window = label_array[y:y+crop_size, x:x+crop_size]
        target_pixels = np.sum(window == 1)
        # total_pixels = crop_size * crop_size

        # 检查条件：至少包含目标类且占比≥50%
        if target_pixels > 0 and target_pixels / total_pixels >= 0.75:
            valid_windows.append((x, y))
        
        attempts += 1

    if len(valid_windows) < num_samples:
        print(f"Warning: Only found {len(valid_windows)} valid windows for {label_tif}")
    
    return valid_windows[:num_samples]

def crop_and_save(image_tif, label_tif, valid_windows, image_output_dir, label_output_dir, crop_size):
    """根据有效窗口坐标执行裁剪"""
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_tif))[0]

    with rasterio.open(image_tif) as img_src, rasterio.open(label_tif) as lbl_src:
        for idx, (x, y) in enumerate(valid_windows):
            # 创建裁剪窗口
            window = Window(x, y, crop_size, crop_size)
            
            # 读取数据（确保窗口在图像范围内）
            img_crop = img_src.read(window=window, boundless=True, fill_value=0)
            lbl_crop = lbl_src.read(window=window, boundless=True, fill_value=0)

            # 更新元数据
            transform = rasterio.windows.transform(window, img_src.transform)
            img_meta = img_src.meta.copy()
            img_meta.update({
                "width": crop_size,
                "height": crop_size,
                "transform": transform
            })
            lbl_meta = lbl_src.meta.copy()
            lbl_meta.update({
                "width": crop_size,
                "height": crop_size,
                "transform": transform
            })

            # 保存文件
            img_path = os.path.join(image_output_dir, f"{base_name}_randcrop_{idx}.tif")
            lbl_path = os.path.join(label_output_dir, f"{base_name}_randcrop_{idx}.tif")
            
            with rasterio.open(img_path, "w", **img_meta) as img_dst:
                img_dst.write(img_crop)
            with rasterio.open(lbl_path, "w", **lbl_meta) as lbl_dst:
                lbl_dst.write(lbl_crop)

            print(f"Saved: {img_path}, {lbl_path}")

def process_data(image_dir, label_dir, image_output_dir, label_output_dir, crop_size):
    """处理整个数据集"""
    for filename in os.listdir(label_dir):
        if filename.endswith(".tif"):
            label_path = os.path.join(label_dir, filename)
            image_path = os.path.join(image_dir, filename)
            
            if not os.path.exists(image_path):
                continue

            # 生成有效窗口并裁剪
            windows = generate_valid_windows(label_path, crop_size)
            if windows:
                crop_and_save(image_path, label_path, windows, image_output_dir, label_output_dir, crop_size)

# 使用示例
image_dir = "DATA/rawdata/sentinel_nbands"
label_dir = "DATA/rawdata/LabeledTIFF"
image_output_dir = "DATA/crop_400/images"
label_output_dir = "DATA/crop_400/labels"
crop_size = 400

process_data(image_dir, label_dir, image_output_dir, label_output_dir, crop_size)
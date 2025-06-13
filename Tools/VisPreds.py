import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def visualize_binary_results(label_dir, result_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    label_paths = sorted(glob(os.path.join(label_dir, '*.png')))
    result_paths = sorted(glob(os.path.join(result_dir, '*.png')))

    assert len(label_paths) == len(result_paths), "标签图像和预测图像数量不一致"

    for label_path, result_path in tqdm(zip(label_paths, result_paths), total=len(label_paths)):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)

        if label.shape != result.shape:
            raise ValueError(f"Shape mismatch: {label_path} vs {result_path}")

        # 二值化确保0和1
        label_bin = (label > 0).astype(np.uint8)
        result_bin = (result > 0).astype(np.uint8)

        tp = (label_bin == 1) & (result_bin == 1)
        tn = (label_bin == 0) & (result_bin == 0)
        fp = (label_bin == 0) & (result_bin == 1)
        fn = (label_bin == 1) & (result_bin == 0)

        # 初始化可视化图像（RGB）
        vis = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

        vis[tp] = [255, 255, 255]   # 白色
        vis[fp] = [0, 0, 255]       # 红色
        vis[fn] = [0, 255, 0]       # 绿色
        vis[tn] = [0, 0, 0]         # 黑色

        # 保存
        basename = os.path.basename(label_path)
        save_path = os.path.join(save_dir, basename)
        cv2.imwrite(save_path, vis)

    print(f"完成，共处理 {len(label_paths)} 张图像。结果保存在：{save_dir}")

if __name__ == '__main__':
    label_dir = 'DATA/crop_400/subset/labels_vis'
    result_dir = 'DATA/crop_400/subset/deeplab_vis'
    save_dir = 'DATA/crop_400/subset/deeplab_vis_fp'
    visualize_binary_results(label_dir, result_dir, save_dir)
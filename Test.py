import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Models.unet_resnet import Unet


class RSDataset(Dataset):
    def __init__(self, msi_dir):
        self.msi_dir = msi_dir
        self.file_list = [f for f in os.listdir(msi_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        msi_path = os.path.join(self.msi_dir, self.file_list[idx])

        with rasterio.open(msi_path) as msi_ds:
            msi = msi_ds.read().astype(np.float32)
        img = np.nan_to_num(msi) / 10000.0

        # Sentinel-2 波段索引
        B2, B3, B4, B8, B11 = img[1], img[2], img[3], img[7], img[10]

        eps = 1e-6  # 避免除0

        # 计算遥感指数
        ndvi = (B8 - B4) / (B8 + B4 + eps)
        evi = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + eps)
        ndwi = (B3 - B8) / (B3 + B8 + eps)
        ndbi = (B11 - B8) / (B11 + B8 + eps)
        ndsi = (B3 - B11) / (B3 + B11 + eps)
        bsi = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + eps)

        ndvi = np.clip(ndvi, -1, 1)
        evi = np.clip(evi, -1, 2.5)
        ndwi = np.clip(ndwi, -1, 1)
        ndbi = np.clip(ndbi, -1, 1)
        ndsi = np.clip(ndsi, -1, 1)
        bsi = np.clip(bsi, -1, 1)

        # 合并为新通道
        indices = np.stack([ndvi, evi, ndwi, ndbi, ndsi, bsi], axis=0)

        # 拼接原始图像和指数通道
        img = np.concatenate([img, indices], axis=0)

        # 去除 nan 值
        img = np.nan_to_num(img)

        return img, msi_path

def ISW(model, input_data, window_size=(256, 256), overlap=0, device=None):
    _, _, height, width = input_data.size()
    stride = int(window_size[0] * (1 - overlap))

    # Initialize an empty output array
    output_array = np.zeros((1, height, width), dtype=np.uint8)

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Adjust window to stay within image boundaries
            if y + window_size[0] > height:
                y_start = height - window_size[0]
            else:
                y_start = y
            if x + window_size[1] > width:
                x_start = width - window_size[1]
            else:
                x_start = x

            # Extract a window from the input data
            window = input_data[:, :, y_start:y_start + window_size[0], x_start:x_start + window_size[1]]

            with torch.no_grad():
                # Perform inference on the window
                output_window = model(window.to(device))

            # Place the output back into the output array
            output_array[:, y_start:y_start + window_size[0], x_start:x_start + window_size[1]] = np.argmax(output_window.cpu().numpy(), axis=1)[0]

    return output_array

def main():
    # ==== 所有参数在此处定义 ====
    msi_dir = 'DATA/rawdata/sentinel_nbands'
    model_path = 'CODE/dl4rs/features_21/logs/unet/Unet_MMSegYREB.pth'
    output_dir = 'unet_results'
    device = 'cuda'  # 'cuda' or 'cpu'
    use_dp = False  # 使用 DataParallel 多卡并行
    num_classes = 2
    windows_size = (400, 400)  # 窗口大小
    # ============================

    os.makedirs(output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dataset = RSDataset(msi_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Unet(num_classes=num_classes, backbone='resnet50', in_channels=21)
    if use_dp:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model.to(device)
    model.eval()

    for msi, paths in tqdm(dataloader):

        msi = msi.to(device)

        output_array = ISW(model, msi, device=device, window_size=windows_size)

        _, filename = os.path.split(paths[0])
        output_path = os.path.join(output_dir, f'{filename}')

        # Read metadata using rasterio
        with rasterio.open(paths[0]) as src:
            transform = src.transform
            profile = src.profile
            crs = src.crs

        # Update profile for output file
        profile.update(
            dtype=rasterio.uint8,
            count=1,
        )

        # Save the result using rasterio
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_array[0], 1)

    print("Inference completed.")

if __name__ == '__main__':
    main()
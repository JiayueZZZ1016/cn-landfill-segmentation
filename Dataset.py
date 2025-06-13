import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import rasterio
from tqdm import tqdm

class RSDataset(Dataset):

    CLASSES = [
        'background', 'landfills'
    ]

    def __init__(self, root: str, split: str = 'train') -> None:
        assert split in ['train', 'test'], f"Invalid split: {split}"
        self.root = Path(root)
        self.split = split
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        print(f"Loading {split} data...")
        self.msi_paths, self.label_paths = self._gather_files()
        print(f"Found {len(self.msi_paths)} {split} samples.")

    def _gather_files(self) -> Tuple[List[Path], List[List[Path]]]:
        label_dir = self.root / 'labels'
        msi_dir = self.root / 'images'
        label_files = list(label_dir.rglob('*.tif'))

        msi_paths, label_paths = [], []
        file_list = (self.root / f"{self.split}.txt").read_text().splitlines()

        for filename in tqdm(file_list):
            name = filename.rsplit('.', 1)[0]
            msi_paths.append(msi_dir / filename)

            matched_labels = [f for f in label_files if f.stem.startswith(name)]
            assert matched_labels, f"No label found for {name}"
            label_paths.append(matched_labels)

        assert len(msi_paths) == len(label_paths)
        return msi_paths, label_paths

    def __len__(self) -> int:
        return len(self.msi_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        msi = self._read_image_2(self.msi_paths[index], scale=10000.0)
        label = self._read_label_stack(self.label_paths[index])
        return msi, label.squeeze().long()

    def _read_image(self, path: Path, scale: float) -> Tensor:
        with rasterio.open(str(path)) as src:
            img = src.read().astype(np.float32)
        img = np.nan_to_num(img) / scale
        return torch.from_numpy(img).float()
    
    def _read_image_2(self, path: Path, scale: float) -> Tensor:
        with rasterio.open(str(path)) as src:
            img = src.read().astype(np.float32)
        img = np.nan_to_num(img) / scale

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

        return torch.from_numpy(img).float()

    def _read_label_stack(self, paths: List[Path]) -> Tensor:
        labels, mask = None, None
        for path in paths:
            with rasterio.open(str(path)) as src:
                lbl = src.read(1).astype(np.uint8)
            if mask is None:
                mask = np.zeros_like(lbl, dtype=np.uint8)
            lbl_masked = np.ma.masked_array(lbl, mask=mask)
            mask += np.minimum(lbl, 1)
            labels = lbl_masked if labels is None else labels + lbl_masked
        return torch.from_numpy(labels.data).unsqueeze(0).to(torch.uint8)

if __name__ == '__main__':
    dataset = RSDataset('DATA/crop_400', split='train')
    for i in range(len(dataset)):
        msi, label = dataset[i]
        print(f"Sample {i}: MSI {msi.shape}, Label {label.shape}, {label.unique()}")
import os
import torch
import warnings
from torch import nn
from tqdm import tqdm
from tabulate import tabulate
from torch.backends import cudnn
from torch.utils.data import DataLoader

from Models.unet_resnet import Unet
from Dataset import RSDataset
from Metrics import Metrics


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def setup_cudnn() -> None:
    cudnn.benchmark = True
    cudnn.deterministic = False

@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()
    metrics = Metrics(dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device)

    for msi, labels in tqdm(dataloader):
        msi = msi.to(device)
        labels = labels.to(device)
        preds = model(msi).softmax(dim=1)
        metrics.update(preds, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    oa = metrics.compute_oa()
    recall, mrecall = metrics.compute_recall()

    return acc, macc, f1, mf1, ious, miou, oa, recall, mrecall


def main():
    # ✅ 推理参数手动指定
    device = torch.device("cuda")
    dataset_root = "DATA/crop_400"
    batch_size = 32
    model_path = "CODE/dl4rs/features_21/logs/unet/Unet_MMSegYREB.pth"

    # ✅ 数据加载
    dataset = RSDataset(dataset_root, 'test')
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # ✅ 模型构建与加载
    model = Unet(num_classes=2, backbone='resnet50', in_channels=21).to(device)
    use_dp = False
    if use_dp:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model = model.to(device)
    
    # ✅ 开始评估
    acc, macc, f1, mf1, ious, miou, oa, recall, mrecall = evaluate(model, dataloader, device)

    # ✅ 输出结果
    table = {
        'Class': list(dataset.CLASSES) + ['Mean'],
        'IoU': ious + [miou],
        'F1': f1 + [mf1],
        'Precision': acc + [macc],
        'Recall': recall + [mrecall]
    }

    print(tabulate(table, headers='keys'))
    print(f"Overall Accuracy (OA): {oa:.3f}")
    print(f"Mean IoU: {miou:.3f}")
    print(f"Mean F1 Score: {mf1:.3f}")


if __name__ == '__main__':
    setup_cudnn()
    main()
import os
import time
import torch
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
from torch.backends import cudnn
from tabulate import tabulate
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler, autocast

from Models.unet_resnet import Unet
from Dataset import RSDataset
from Losses import *
from Schedulers import get_scheduler
from Optimizers import get_optimizer
from Val import evaluate

# nohup python CODE/dl4rs/landfill/Train.py > CODE/dl4rs/landfill/logs/unet/nohup.txt 2>&1 &echo $! >> CODE/dl4rs/landfill/logs/unet/nohup.txt

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 

def log_metrics(oa, miou, macc, mf1, mrecall, log_file):
    with open(log_file, 'a') as f:
        f.write(f"Current OA: {oa:.3f}, mIoU: {miou:.3f}, mAcc: {macc:.3f}, mF1: {mf1:.3f}, mRecall: {mrecall:.3f}\n")

def fix_seeds(seed: int = 3407) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_cudnn() -> None:
    cudnn.benchmark = True
    cudnn.deterministic = False

def main():
    # ✅ 训练配置
    device = torch.device("cuda")
    epochs = 100
    batch_size = 32
    num_workers = 4
    amp_enabled = True
    eval_interval = 1

    # ✅ 保存配置
    save_dir = Path("CODE/dl4rs/landfill/logs/unet")
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "log.txt"

    # ✅ 数据路径和数据集配置
    dataset_root = "DATA/crop_400"
    ignore_label = 255

    # ✅ 模型配置
    model_name = "Unet"
    use_dp = False 
    model = Unet(num_classes=2, backbone='resnet50', in_channels=21).to(device)

    if use_dp:
        model = nn.DataParallel(model)

    # ✅ 损失函数配置
    loss_1 = OhemCrossEntropy(ignore_label=ignore_label).to(device)
    loss_2 = DiceLoss(ignore_label=ignore_label).to(device)
    loss_3 = FocalLoss(ignore_label=ignore_label).to(device)
    loss_fn = [loss_1, loss_2, loss_3]

    # ✅ 优化器配置
    optimizer_name = "adamw"
    learning_rate = 0.001
    weight_decay = 0.01
    scheduler_name = "warmuppolylr"
    scheduler_power = 0.9
    warmup_iters = 10
    warmup_ratio = 0.1

    # ✅ 初始化
    fix_seeds(3407)
    setup_cudnn()

    # ✅ 构建数据集与加载器
    trainset = RSDataset(dataset_root, "train")
    valset =RSDataset(dataset_root, "test")

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=RandomSampler(trainset))
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    # ✅ 构建模型，优化器，学习率调度器
    optimizer = get_optimizer(model, optimizer_name, learning_rate, weight_decay)
    iters_per_epoch = len(trainloader)
    scheduler = get_scheduler(scheduler_name, optimizer, epochs * iters_per_epoch, scheduler_power, iters_per_epoch * warmup_iters, warmup_ratio)
    scaler = GradScaler(enabled=amp_enabled)

    best_miou = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch,
                    desc=f"Epoch: [{epoch+1}/{epochs}]")

        for i, (msi, lbl) in pbar:
            msi, lbl = msi.to(device), lbl.to(device)
            optimizer.zero_grad()

            with autocast(enabled=amp_enabled):
                logits = model(msi)
                loss = sum(fn(logits, lbl) for fn in loss_fn)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            total_loss += loss.item()
            current_lr = sum(scheduler.get_lr()) / len(scheduler.get_lr())
            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{i+1}/{iters_per_epoch}] "
                                 f"LR: {current_lr:.6f} Loss: {total_loss/(i+1):.4f}")

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs:
            acc, macc, f1, mf1, ious, miou, oa, recall, mrecall = evaluate(model, valloader, device)

            log_metrics(oa, miou, macc, mf1, mrecall, log_file)

            # ✅ 输出结果
            table = {
                'Class': list(trainset.CLASSES) + ['Mean'],
                'IoU': ious + [miou],
                'F1': f1 + [mf1],
                'Precision': acc + [macc],
                'Recall': recall + [mrecall]
            }

            print(tabulate(table, headers='keys'))
            print(f"Overall Accuracy (OA): {oa:.3f}")
            print(f"Mean IoU: {miou:.3f}")
            print(f"Mean F1 Score: {mf1:.3f}")

            print(f"Current mIoU: {miou:.3f}, Best mIoU: {best_miou:.3f}")

            if miou > best_miou and all(iou != 0 for iou in ious):
                best_miou = miou
                save_path = save_dir / f"{model_name}_MMSegYREB.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            

    elapsed = time.gmtime(time.time() - start_time)
    print(tabulate([
        ['Best mIoU', f"{best_miou:.2f}"],
        ['Training Time', time.strftime("%H:%M:%S", elapsed)]
    ]))


if __name__ == '__main__':
    main()
"""
Advanced segmentation training with configurable seed and augmentation.
Usage:
    python train_advanced.py --seed 42 --aug A
    python train_advanced.py --seed 43 --aug B
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ========== Parse arguments ==========
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--aug', type=str, choices=['A', 'B'], default='A',
                    help='Augmentation policy: A (basic) or B (with rotation/cutout)')
args = parser.parse_args()

# ========== Configuration ==========
SEED = args.seed
AUG_POLICY = args.aug
BACKBONE_SIZE = "small"
BATCH_SIZE = 2
LR = 1e-4
N_EPOCHS = 35
IMAGE_SIZE = (540, 960)   # original
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths (adjust if needed)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(SCRIPT_DIR, 'Offroad_Segmentation_Training_Dataset', 'train')
VAL_DIR   = os.path.join(SCRIPT_DIR, 'Offroad_Segmentation_Training_Dataset', 'val')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'train_stats_seed{SEED}_{AUG_POLICY}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Seed everything ==========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ========== Mask conversion ==========
value_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
n_classes = len(value_map)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)

# ========== Compute target size (multiple of 14) ==========
h, w = IMAGE_SIZE
h = int(((h // 2) // 14) * 14)   # 266
w = int(((w // 2) // 14) * 14)   # 476
print(f"Resized to: {h} x {w}")

# ========== Augmentation pipelines ==========
# Policy A: horizontal flip + color jitter + blur (no rotation)
if AUG_POLICY == 'A':
    train_transform = A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.7),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
# Policy B: add rotation and cutout (fixed to preserve dimensions)
else:  # B
    train_transform = A.Compose([
        A.Resize(h, w),
        A.Rotate(limit=45, p=0.5),                
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.CoarseDropout(num_holes=8, max_h=20, max_w=20, fill_value=0, p=0.2),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

val_transform = A.Compose([
    A.Resize(h, w),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])

# ========== Dataset class ==========
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)
        image = np.array(image)
        mask = np.array(mask)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask.long(), data_id

# ========== Loaders ==========
trainset = MaskDataset(TRAIN_DIR, transform=train_transform)
valset   = MaskDataset(VAL_DIR,   transform=val_transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train: {len(trainset)}  Val: {len(valset)}")

# ========== DINOv2 backbone ==========
backbone_archs = {"small":"vits14","base":"vitb14_reg","large":"vitl14_reg","giant":"vitg14_reg"}
backbone_name = f"dinov2_{backbone_archs[BACKBONE_SIZE]}"
print(f"Loading {backbone_name} ...")
backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
backbone.eval()
backbone.to(DEVICE)
for p in backbone.parameters():
    p.requires_grad = False

# Get embedding dim
sample_img, _, _ = trainset[0]
sample_img = sample_img.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    feat = backbone.forward_features(sample_img)["x_norm_patchtokens"]
embed_dim = feat.shape[2]
print(f"Embedding dim: {embed_dim}")

# ========== Segmentation head ==========
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0,3,1,2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

tokenH, tokenW = h//14, w//14
model = SegmentationHeadConvNeXt(embed_dim, n_classes, tokenW, tokenH).to(DEVICE)

# ========== Loss functions ==========
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0,3,1,2).float()
        intersection = (probs * targets_onehot).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

criterion_ce = nn.CrossEntropyLoss()
criterion_dice = DiceLoss()

# ========== Optimizer & scheduler ==========
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
scaler = GradScaler('cuda')      # correct AMP scaler

# ========== Metrics ==========
def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    ious = []
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append((inter/union).item() if union>0 else float('nan'))
    return np.nanmean(ious)

def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    dices = []
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().float()
        dice = (2.*inter + smooth) / (p.sum().float() + t.sum().float() + smooth)
        dices.append(dice.item())
    return np.mean(dices)

def compute_pixel_acc(pred, target):
    pred = torch.argmax(pred, dim=1)
    return (pred == target).float().mean().item()

def evaluate(model, loader, device):
    model.eval()
    ious, dices, accs = [], [], []
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast('cuda'):                     # correct autocast
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
            ious.append(compute_iou(outputs, masks))
            dices.append(compute_dice(outputs, masks))
            accs.append(compute_pixel_acc(outputs, masks))
    return np.mean(ious), np.mean(dices), np.mean(accs)

# ========== History ==========
history = {k:[] for k in ['train_loss','val_loss','train_iou','val_iou','train_dice','val_dice','train_acc','val_acc','lr']}

# ========== Training loop ==========
def main():
    print("Starting training...")
    for epoch in range(N_EPOCHS):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]")
        for imgs, masks, _ in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with autocast('cuda'):                     # correct autocast
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion_ce(outputs, masks) + criterion_dice(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with autocast('cuda'):                 # correct autocast
                    feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                    logits = model(feats)
                    outputs = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                    loss = criterion_ce(outputs, masks) + criterion_dice(outputs, masks)
                val_losses.append(loss.item())

        # Metrics
        train_iou, train_dice, train_acc = evaluate(model, train_loader, DEVICE)
        val_iou,   val_dice,   val_acc   = evaluate(model, val_loader, DEVICE)

        # Record
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.get_last_lr()[0])

        scheduler.step()
        print(f"Epoch {epoch+1:2d} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Val IoU: {val_iou:.4f} | Val Acc: {val_acc:.4f}")

    # ========== Save model ==========
    model_path = os.path.join(SCRIPT_DIR, f"seg_head_seed{SEED}_{AUG_POLICY}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("Training complete!")

if __name__ == '__main__':
    main()
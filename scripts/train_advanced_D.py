import torch, os, random, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ========= CONFIG =========
SEED = 45
BATCH_SIZE = 1
LR = 1e-4
EPOCHS = 35
DEVICE = torch.device("cuda")

SCRIPT = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(SCRIPT,"Offroad_Segmentation_Training_Dataset","train")
VAL   = os.path.join(SCRIPT,"Offroad_Segmentation_Training_Dataset","val")

# ========= SEED =========
def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)

# ========= MASK MAP =========
value_map = {0:0,100:1,200:2,300:3,500:4,550:5,700:6,800:7,7100:8,10000:9}
NCLS = len(value_map)

def convert(mask):
    arr = np.array(mask)
    out = np.zeros_like(arr)
    for k,v in value_map.items(): out[arr==k] = v
    return out

# ========= DATASET =========
class DS(Dataset):
    def __init__(self, root, tf):
        self.img = os.path.join(root,"Color_Images")
        self.msk = os.path.join(root,"Segmentation")
        self.ids = sorted(os.listdir(self.img))
        self.tf = tf
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        im = np.array(Image.open(os.path.join(self.img,self.ids[i])).convert("RGB"))
        mk = convert(Image.open(os.path.join(self.msk,self.ids[i])))
        aug = self.tf(image=im, mask=mk)
        return aug["image"], aug["mask"].long()

H,W = 266,476

train_tf = A.Compose([
    A.Resize(H,W),
    A.Rotate(limit=45,p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(0.3,0.3,0.3,0.2,p=0.7),
    A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ToTensorV2()
])
val_tf = A.Compose([
    A.Resize(H,W),
    A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ToTensorV2()
])

train_dl = DataLoader(DS(TRAIN,train_tf),BATCH_SIZE,shuffle=True,num_workers=0)
val_dl   = DataLoader(DS(VAL,val_tf),BATCH_SIZE,num_workers=0)

# ========= BACKBONE =========
backbone = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14_reg").to(DEVICE)

# unfreeze ONLY last block
for n,p in backbone.named_parameters():
    p.requires_grad = ("blocks.11" in n)

# ========= HEAD =========
class Head(nn.Module):
    def __init__(self,C,classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C,128,7,padding=3), nn.GELU(),
            nn.Conv2d(128,128,7,padding=3,groups=128), nn.GELU(),
            nn.Conv2d(128,128,1), nn.GELU(),
            nn.Conv2d(128,classes,1)
        )
    def forward(self,x):
        B,N,C = x.shape
        x = x.reshape(B,H//14,W//14,C).permute(0,3,1,2)
        return self.net(x)

with torch.no_grad():
    sample = next(iter(train_dl))[0].to(DEVICE)
    C = backbone.forward_features(sample)["x_norm_patchtokens"].shape[2]

model = Head(C,NCLS).to(DEVICE)

# ========= LOSS =========
ce = nn.CrossEntropyLoss()
class Dice(nn.Module):
    def forward(self, logits, target, smooth=1e-6):
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target, num_classes=probs.shape[1]).permute(0,3,1,2).float()

        inter = (probs * onehot).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + onehot.sum(dim=(2,3))

        dice = (2 * inter + smooth) / (union + smooth)

        return 1 - dice.mean()

dice = Dice()

# ========= OPT =========
opt = optim.AdamW([
    {"params":model.parameters(),"lr":1e-4},
    {"params":[p for p in backbone.parameters() if p.requires_grad],"lr":1e-5}
],weight_decay=1e-4)

sched = optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
scaler = GradScaler("cuda")

# ========= IOU =========
def val_iou():
    model.eval()
    inter=union=0
    with torch.no_grad():
        for x,y in val_dl:
            x,y=x.to(DEVICE),y.to(DEVICE)
            with autocast("cuda"):
                f=backbone.forward_features(x)["x_norm_patchtokens"]
                o=F.interpolate(model(f),size=x.shape[2:],mode="bilinear")
            p=o.argmax(1)
            inter += (p==y).sum().item()
            union += y.numel()
    return inter/union

# ========= TRAIN =========
best=0
for ep in range(EPOCHS):
    model.train(); backbone.train()
    loop=tqdm(train_dl,desc=f"Ep {ep+1}/{EPOCHS}")
    for x,y in loop:
        x,y=x.to(DEVICE),y.to(DEVICE)
        opt.zero_grad()
        with autocast("cuda"):
            f=backbone.forward_features(x)["x_norm_patchtokens"]
            o=F.interpolate(model(f),size=x.shape[2:],mode="bilinear")
            loss = 0.5*ce(o,y)+0.5*dice(o,y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        loop.set_postfix(loss=float(loss))

    iou = val_iou()
    print("VAL IoU:",iou)
    if iou>best:
        best=iou
        torch.save(model.state_dict(),"model_D_best.pth")

    sched.step()
    torch.cuda.empty_cache(); gc.collect()

print("BEST IoU:",best)

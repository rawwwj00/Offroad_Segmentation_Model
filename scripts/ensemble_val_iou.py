import torch, os
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"C:\Users\rajjs\Desktop\Startathon\new_"
VAL = os.path.join(ROOT, "Offroad_Segmentation_Training_Dataset", "val")

def resize_to_patch_multiple(x, scale, patch=14):
    B, C, H, W = x.shape

    newH = int(round(H * scale))
    newW = int(round(W * scale))

    # snap to nearest multiple of patch size
    newH = (newH // patch) * patch
    newW = (newW // patch) * patch

    return torch.nn.functional.interpolate(
        x, size=(newH, newW), mode="bilinear", align_corners=False
    )


# ---------- label map ----------
value_map = {0:0,100:1,200:2,300:3,500:4,550:5,700:6,800:7,7100:8,10000:9}
NCLS = len(value_map)

def convert(mask):
    arr = np.array(mask)
    out = np.zeros_like(arr)
    for k,v in value_map.items():
        out[arr==k] = v
    return out

# ---------- dataset ----------
class DS(Dataset):
    def __init__(self):
        self.img = os.path.join(VAL,"Color_Images")
        self.msk = os.path.join(VAL,"Segmentation")
        self.ids = sorted(os.listdir(self.img))

        self.tf = A.Compose([
            A.Resize(266,476),
            A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ToTensorV2()
        ])

    def __len__(self): return len(self.ids)

    def __getitem__(self,i):
        im = np.array(Image.open(os.path.join(self.img,self.ids[i])).convert("RGB"))
        mk = convert(Image.open(os.path.join(self.msk,self.ids[i])))
        aug = self.tf(image=im, mask=mk)
        return aug["image"], aug["mask"]

loader = DataLoader(DS(), batch_size=2, shuffle=False)

# ---------- model weights ----------
WEIGHTS = {
    "A": 0.469,
    "B": 0.462,
    "C": 0.462,
    "D": 0.348,
    "E": 0.293,
}

MODEL_PATHS = {
    "A": os.path.join(ROOT, "model_A_best.pth"),
    "B": os.path.join(ROOT, "model_B_best.pth"),
    "C": os.path.join(ROOT, "model_C_best.pth"),
    "D": os.path.join(ROOT, "model_D_best.pth"),
    "E": os.path.join(ROOT, "model_E_best.pth"),
}

s = sum(WEIGHTS.values())
WEIGHTS = {k: v/s for k, v in WEIGHTS.items()}
# ---------- backbone ----------
backbone = torch.hub.load("facebookresearch/dinov2","dinov2_vits14").to(DEVICE).eval()

class Head(torch.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(C, 128, 7, padding=3),
            torch.nn.GELU()
        )
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 7, padding=3, groups=128),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, 128, 1),
            torch.nn.GELU()
        )
        self.classifier = torch.nn.Conv2d(128, NCLS, 1)

    def forward(self, x, H, W):
        """
        x: (B, N, C)
        H, W: token grid size
        """
        B, N, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ---------- embed dim ----------
dummy = torch.randn(1,3,266,476).to(DEVICE)
C = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]

# ---------- load models ----------
models = {}
for name,path in MODEL_PATHS.items():
    m = Head(C).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    models[name] = m

# ---------- IoU computation ----------
def mean_iou(pred, target, ncls=10):
    ious = []
    for cls in range(ncls):
        p = (pred == cls)
        t = (target == cls)
        union = (p | t).sum()
        if union > 0:
            ious.append((p & t).sum() / union)
    return np.mean(ious)

# ---------- TTA settings ----------
SCALES = [1.0, 1.25]
FLIPS = [False, True]

# ---------- evaluate ----------
ious = []

with torch.no_grad():
    for x,y in loader:

        x = x.to(DEVICE)
        y = y.numpy()

        B,_,H0,W0 = x.shape
        ensemble_logits = torch.zeros(B, NCLS, H0, W0).to(DEVICE)
        count = 0

        for scale in SCALES:

            xs = resize_to_patch_multiple(x, scale)

            for flip in FLIPS:

                inp = torch.flip(xs, [3]) if flip else xs
                feats = backbone.forward_features(inp)["x_norm_patchtokens"]

                logits_sum = 0
                for name,model in models.items():
                    B, N, C = feats.shape
                    H = int(np.sqrt(N * 266 / 476))
                    W = N // H

                    lg = model(feats, H, W)

                    lg = F.interpolate(lg, size=inp.shape[2:], mode="bilinear", align_corners=False)
                    logits_sum += WEIGHTS[name] * lg

                if flip:
                    logits_sum = torch.flip(logits_sum, [3])

                logits_sum = F.interpolate(logits_sum, size=(H0,W0), mode="bilinear", align_corners=False)

                ensemble_logits += logits_sum
                count += 1

        pred = (ensemble_logits / count).argmax(1).cpu().numpy()

        for p,t in zip(pred, y):
            ious.append(mean_iou(p, t))

print(f"\n🔥 Ensemble Validation IoU: {np.mean(ious):.4f}")

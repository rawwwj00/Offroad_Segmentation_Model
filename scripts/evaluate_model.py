import torch, os
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"C:\Users\rajjs\Desktop\Startathon\new_"
VAL  = os.path.join(ROOT, "Offroad_Segmentation_Training_Dataset", "val")

# ---------- mask map ----------
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
    def __init__(self, root):
        self.img = os.path.join(root,"Color_Images")
        self.msk = os.path.join(root,"Segmentation")
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
        return aug["image"], aug["mask"].long()

val_dl = DataLoader(DS(VAL), batch_size=2, shuffle=False)

# ---------- TRUE IoU ----------
def compute_iou(backbone, head):
    backbone.eval()
    head.eval()
    ious = []

    with torch.no_grad():
        for x,y in val_dl:
            x,y = x.to(DEVICE), y.to(DEVICE)

            feats = backbone.forward_features(x)["x_norm_patchtokens"]
            logits = head(feats)
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

            pred = logits.argmax(1)

            for cls in range(NCLS):
                p = (pred == cls)
                t = (y == cls)
                union = (p | t).sum().float()
                if union > 0:
                    ious.append(((p & t).sum().float() / union).item())

    return np.mean(ious)

# ---------- load model correctly ----------
def load_model(path):

    state = torch.load(path, map_location=DEVICE)

    # detect embed dim
    if "net.0.weight" in state:
        embed_dim = state["net.0.weight"].shape[1]
        head_type = "B"
    else:
        embed_dim = state["stem.0.weight"].shape[1]
        head_type = "A"

    # choose correct backbone
    if embed_dim == 384:
        backbone = torch.hub.load("facebookresearch/dinov2","dinov2_vits14").to(DEVICE)
    elif embed_dim == 768:
        backbone = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14_reg").to(DEVICE)
    else:
        raise RuntimeError("Unknown embed dim")

    # ----- head A -----
    if head_type == "A":

        class HeadA(torch.nn.Module):
            def __init__(self, C):
                super().__init__()
                self.H, self.W = 266//14, 476//14

                self.stem = torch.nn.Sequential(
                    torch.nn.Conv2d(C,128,7,padding=3), torch.nn.GELU()
                )

                self.block = torch.nn.Sequential(
                    torch.nn.Conv2d(128,128,7,padding=3,groups=128), torch.nn.GELU(),
                    torch.nn.Conv2d(128,128,1), torch.nn.GELU()
                )

                self.classifier = torch.nn.Conv2d(128,NCLS,1)

            def forward(self,x):
                B,N,C = x.shape
                x = x.reshape(B,self.H,self.W,C).permute(0,3,1,2)
                x = self.stem(x)
                x = self.block(x)
                return self.classifier(x)

        head = HeadA(embed_dim).to(DEVICE)

    # ----- head B -----
    else:

        class HeadB(torch.nn.Module):
            def __init__(self, C):
                super().__init__()
                self.H, self.W = 266//14, 476//14
                self.net = torch.nn.Sequential(
                    torch.nn.Conv2d(C,128,7,padding=3), torch.nn.GELU(),
                    torch.nn.Conv2d(128,128,7,padding=3,groups=128), torch.nn.GELU(),
                    torch.nn.Conv2d(128,128,1), torch.nn.GELU(),
                    torch.nn.Conv2d(128,NCLS,1)
                )

            def forward(self,x):
                B,N,C = x.shape
                x = x.reshape(B,self.H,self.W,C).permute(0,3,1,2)
                return self.net(x)

        head = HeadB(embed_dim).to(DEVICE)

    head.load_state_dict(state)
    return backbone, head

# ---------- paths ----------
models = {
    "A": os.path.join(ROOT,"model_A_best.pth"),
    "B": os.path.join(ROOT,"model_B_best.pth"),
    "C": os.path.join(ROOT,"model_C_best.pth"),
    "D": os.path.join(ROOT,"model_D_best.pth"),
    "E": os.path.join(ROOT,"model_E_best.pth"),
}

# ---------- run ----------
for name, path in models.items():

    if not os.path.exists(path):
        print(f"{name}: model not found")
        continue

    backbone, head = load_model(path)
    iou = compute_iou(backbone, head)

    print(f"{name} REAL IoU: {iou:.4f}")

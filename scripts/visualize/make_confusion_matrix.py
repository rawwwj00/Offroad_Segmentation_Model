import torch, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = r"C:\Users\rajjs\Desktop\Startathon\new_"
VAL = os.path.join(ROOT, "Offroad_Segmentation_Training_Dataset", "val")

value_map = {0:0,100:1,200:2,300:3,500:4,550:5,700:6,800:7,7100:8,10000:9}
NCLS = 10

def convert(mask):
    arr = np.array(mask)
    out = np.zeros_like(arr)
    for k,v in value_map.items():
        out[arr==k] = v
    return out

class DS(torch.utils.data.Dataset):
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

loader = DataLoader(DS(), batch_size=2)

# -------- load best model A (simplest) --------
backbone = torch.hub.load("facebookresearch/dinov2","dinov2_vits14").to(DEVICE).eval()

class Head(torch.nn.Module):
    def __init__(self,C):
        super().__init__()
        self.H,self.W = 19,34
        self.stem = torch.nn.Sequential(torch.nn.Conv2d(C,128,7,padding=3),torch.nn.GELU())
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(128,128,7,padding=3,groups=128),torch.nn.GELU(),
            torch.nn.Conv2d(128,128,1),torch.nn.GELU())
        self.classifier = torch.nn.Conv2d(128,NCLS,1)

    def forward(self,x):
        B,N,C = x.shape
        x = x.reshape(B,self.H,self.W,C).permute(0,3,1,2)
        return self.classifier(self.block(self.stem(x)))

dummy = torch.randn(1,3,266,476).to(DEVICE)
C = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]

head = Head(C).to(DEVICE)
head.load_state_dict(torch.load(os.path.join(ROOT,"model_A_best.pth"), map_location=DEVICE))
head.eval()

# -------- collect predictions --------
y_true, y_pred = [], []

with torch.no_grad():
    for x,y in loader:
        x = x.to(DEVICE)
        feats = backbone.forward_features(x)["x_norm_patchtokens"]
        logits = head(feats)
        logits = torch.nn.functional.interpolate(logits, size=(266,476))
        pred = logits.argmax(1).cpu().numpy()

        y_true.extend(y.numpy().reshape(-1))
        y_pred.extend(pred.reshape(-1))

cm = confusion_matrix(y_true, y_pred, labels=list(range(NCLS)))

plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(ROOT,"confusion_matrix.png"))
plt.close()

print("Saved confusion_matrix.png")

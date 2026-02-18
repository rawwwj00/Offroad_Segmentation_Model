import torch, os
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================= DEVICE =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ================= PATHS =================
ROOT = r"C:\Users\rajjs\Desktop\Startathon\new_"
TEST_DIR = os.path.join(ROOT, "Offroad_Segmentation_testImages", "Color_Images")

SAVE_LABEL = os.path.join(ROOT, "submission_masks")
SAVE_COLOR = os.path.join(ROOT, "submission_visual")

os.makedirs(SAVE_LABEL, exist_ok=True)
os.makedirs(SAVE_COLOR, exist_ok=True)

# ================= MODEL WEIGHTS =================
WEIGHTS = {
    "A": 0.348,
    "B": 0.325,
    "C": 0.327,
}

MODEL_PATHS = {
    "A": os.path.join(ROOT, "model_A_best.pth"),
    "B": os.path.join(ROOT, "model_B_best.pth"),
    "C": os.path.join(ROOT, "model_C_best.pth"),
}

NCLS = 10
H, W = 266, 476

# ================= BACKBONE =================
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE)
backbone.eval()

# ================= HEAD (TRAINING MATCH) =================
class Head(torch.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.H, self.W = H // 14, W // 14

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

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# ================= GET EMBED DIM =================
dummy = torch.randn(1, 3, H, W).to(DEVICE)
with torch.no_grad():
    C = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]

# ================= LOAD MODELS =================
models = {}
for name, path in MODEL_PATHS.items():
    head = Head(C).to(DEVICE)
    head.load_state_dict(torch.load(path, map_location=DEVICE))
    head.eval()
    models[name] = head

print("Loaded models:", list(models.keys()))

# ================= PREPROCESS =================
normalize_tf = A.Compose([
    A.Resize(H, W),
    A.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ToTensorV2()
])

# ================= TTA SETTINGS =================
SCALES = [1.0, 0.75, 1.25]
FLIPS = [False, True]

# ================= INFERENCE =================
image_ids = sorted(os.listdir(TEST_DIR))

for img_id in tqdm(image_ids):

    img_path = os.path.join(TEST_DIR, img_id)
    image = np.array(Image.open(img_path).convert("RGB"))
    orig_h, orig_w = image.shape[:2]

    ensemble_logits = None
    count = 0

    for scale in SCALES:

        scaled_img = A.Resize(int(orig_h * scale), int(orig_w * scale))(image=image)["image"]
        x = normalize_tf(image=scaled_img)["image"].unsqueeze(0).to(DEVICE)

        for flip in FLIPS:

            inp = torch.flip(x, dims=[3]) if flip else x

            with torch.no_grad():
                feats = backbone.forward_features(inp)["x_norm_patchtokens"]

                logits_sum = 0
                for name, model in models.items():
                    logits = model(feats)
                    logits = F.interpolate(logits, size=inp.shape[2:], mode="bilinear", align_corners=False)
                    logits_sum += WEIGHTS[name] * logits

                if flip:
                    logits_sum = torch.flip(logits_sum, dims=[3])

                logits_sum = F.interpolate(logits_sum, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

                if ensemble_logits is None:
                    ensemble_logits = logits_sum
                else:
                    ensemble_logits += logits_sum

                count += 1

    # ===== FINAL PRED =====
    final_pred = (ensemble_logits / count).argmax(1).squeeze().cpu().numpy().astype(np.uint8)

    # ===== SAVE SUBMISSION MASK (0-9 labels) =====
    Image.fromarray(final_pred).save(os.path.join(SAVE_LABEL, img_id))

    # ===== SAVE COLOR VISUAL MASK =====
    PALETTE = np.array([
        [0, 0, 0],        # 0 background
        [255, 0, 0],      # 1 red
        [0, 255, 0],      # 2 green
        [0, 0, 255],      # 3 blue
        [255, 255, 0],    # 4 yellow
        [255, 0, 255],    # 5 magenta
        [0, 255, 255],    # 6 cyan
        [255, 128, 0],    # 7 orange
        [128, 0, 255],    # 8 purple
        [0, 128, 255],    # 9 sky blue
    ], dtype=np.uint8)

    color_vis = PALETTE[final_pred]
    Image.fromarray(color_vis).save(os.path.join(SAVE_COLOR, img_id))


print("\n🔥 FINAL SUBMISSION MASKS READY.")
print("Submission folder:", SAVE_LABEL)

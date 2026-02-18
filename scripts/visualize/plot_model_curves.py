import torch
import os
import matplotlib.pyplot as plt

# ========= CONFIG =========
ROOT = r"C:\Users\rajjs\Desktop\Startathon\new_"
CKPT_PATH = os.path.join(ROOT, "checkpoints", "checkpoint_epoch10_seed45.pth")
# ↑ change this to A/B/C/D/E checkpoint

SAVE_PREFIX = os.path.basename(CKPT_PATH).replace(".pth", "")

# ========= LOAD =========
ckpt = torch.load(CKPT_PATH, map_location="cpu")
history = ckpt["history"]

print("Loaded history keys:", history.keys())

# ========= HELPER =========
def plot_curve(train, val, title, ylabel, filename):
    plt.figure()
    plt.plot(train, label="train")
    plt.plot(val, label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ROOT, filename), bbox_inches="tight")
    plt.close()
    print("Saved:", filename)

# ========= PLOTS =========

# 1️⃣ Loss
plot_curve(
    history["train_loss"],
    history["val_loss"],
    "Loss vs Epoch",
    "Loss",
    f"{SAVE_PREFIX}_loss.png"
)

# 2️⃣ IoU
plot_curve(
    history["train_iou"],
    history["val_iou"],
    "IoU vs Epoch",
    "IoU",
    f"{SAVE_PREFIX}_iou.png"
)

# 3️⃣ Dice
plot_curve(
    history["train_dice"],
    history["val_dice"],
    "Dice Score vs Epoch",
    "Dice Score",
    f"{SAVE_PREFIX}_dice.png"
)

# 4️⃣ Pixel Accuracy
plot_curve(
    history["train_acc"],
    history["val_acc"],
    "Pixel Accuracy vs Epoch",
    "Accuracy",
    f"{SAVE_PREFIX}_accuracy.png"
)

print("\n✅ All curves generated successfully.")

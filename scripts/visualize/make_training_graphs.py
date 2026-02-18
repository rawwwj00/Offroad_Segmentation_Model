import torch
import matplotlib.pyplot as plt
import os

ROOT = r"C:\Users\rajjs\Desktop\Startathon\new_"
CKPT = os.path.join(ROOT, "checkpoints")  # folder where checkpoints saved

# pick best checkpoint file manually if needed
ckpt_path = os.path.join(CKPT, os.listdir(CKPT)[-1])

data = torch.load(ckpt_path, map_location="cpu")
history = data["history"]

# ---------- LOSS GRAPH ----------
plt.figure()
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(os.path.join(ROOT, "loss_curve.png"))
plt.close()

# ---------- IoU GRAPH ----------
plt.figure()
plt.plot(history["val_iou"], label="Validation IoU")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Validation IoU vs Epoch")
plt.legend()
plt.savefig(os.path.join(ROOT, "iou_curve.png"))
plt.close()

print("Graphs saved: loss_curve.png, iou_curve.png")

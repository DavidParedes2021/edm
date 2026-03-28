import os
import shutil
import random
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
FRAMES_DIR = "../../data/datasets/ead2020/frames"
BBOX_DIR   = "../../data/datasets/ead2020/gt_bbox"
OUTPUT_DIR = "../classifier_outputs/"
#FRAMES_DIR = r"C:\msc\EAD2020_train\EAD2020_dataType_framesOnly\frames"
#BBOX_DIR   = r"C:\msc\EAD2020_train\EAD2020_dataType_framesOnly\gt_bbox"
#OUTPUT_DIR = "classification_dataset"

# ── Class IDs (0-indexed in the .txt files) ────────────────────────────────────
# specularity=0, saturation=1, artifact=2, blur=3, contrast=4, bubbles=5,
# instrument=6, blood=7
SATURATION_ID = 1
CONTRAST_ID   = 4

# ── FIX 1: Add a "both" class so co-occurrence is not silently discarded ───────
CLASSES = ["saturation", "contrast", "both", "none"]
for split in ["train", "val"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)


def get_label_from_bbox(file_path):
    """Return a single class label that preserves co-occurrence."""
    if not os.path.exists(file_path):
        return "none"

    has_saturation = False
    has_contrast   = False

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id == SATURATION_ID:
                has_saturation = True
            elif class_id == CONTRAST_ID:
                has_contrast = True

    if has_saturation and has_contrast:
        return "both"          # FIX 1: preserve both instead of ignoring contrast
    elif has_saturation:
        return "saturation"
    elif has_contrast:
        return "contrast"
    return "none"


# ── Collect dataset ────────────────────────────────────────────────────────────
data = []
for img_name in os.listdir(FRAMES_DIR):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue
    img_path  = os.path.join(FRAMES_DIR, img_name)
    bbox_path = os.path.join(
        BBOX_DIR,
        img_name.replace(".jpg", ".txt").replace(".png", ".txt")
    )
    label = get_label_from_bbox(bbox_path)
    data.append((img_path, label))

labels = [x[1] for x in data]
counts = Counter(labels)
print("Class distribution:", counts)

# ── Train / val split ──────────────────────────────────────────────────────────
train_data, val_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=labels
)


# ── FIX 2: Oversample minority classes in train split ─────────────────────────
def oversample(dataset, target_count=None):
    by_class = {}
    for item in dataset:
        by_class.setdefault(item[1], []).append(item)

    if target_count is None:
        target_count = max(len(v) for v in by_class.values())

    balanced = []
    for cls, items in by_class.items():
        if len(items) < target_count:
            extras = random.choices(items, k=target_count - len(items))
            balanced.extend(items + extras)
        else:
            balanced.extend(items)
    random.shuffle(balanced)
    return balanced

train_data_balanced = oversample(train_data)
print("Balanced train distribution:",
      Counter(x[1] for x in train_data_balanced))


def copy_data(dataset, split):
    seen = {}   # handle duplicates from oversampling
    for img_path, label in dataset:
        base   = os.path.basename(img_path)
        dest   = os.path.join(OUTPUT_DIR, split, label, base)
        # if file already copied (oversampled duplicate), rename it
        if dest in seen:
            seen[dest] += 1
            name, ext = os.path.splitext(base)
            dest = os.path.join(OUTPUT_DIR, split, label,
                                f"{name}_dup{seen[dest]}{ext}")
        else:
            seen[dest] = 0
        shutil.copy(img_path, dest)

copy_data(train_data_balanced, "train")
copy_data(val_data, "val")
print("Dataset prepared!")


# ── Train ──────────────────────────────────────────────────────────────────────
# FIX 3: Use a larger model backbone
model = YOLO("yolov8m-cls.pt")   # medium instead of nano

model.train(
    data=OUTPUT_DIR,
    epochs=80,
    patience=15,
    imgsz=320,
    batch=32,        # DGX GPUs (A100/H100) have large VRAM, increase batch size
    device=0,
    workers=8,       # rule of thumb: 2–4× number of CPU cores per GPU, cap at 8–16

    # FIX 6: Freeze backbone for the first few epochs, then unfreeze
    #freeze=10,            # freeze first 10 layers during warm-up epochs

    # FIX 7: Disable colour-jitter augmentations that corrupt the class signal.
    # Saturation/contrast ARE the signal — don't randomly alter them.
    hsv_s=0.0,            # no saturation jitter
    hsv_v=0.0,            # no brightness jitter
    hsv_h=0.0,            # no hue jitter
    fliplr=0.5,
    flipud=0.1,
    degrees=10,
    translate=0.1,
    scale=0.3,
    lr0=1e-3,
    lrf=0.01,
    warmup_epochs=5,
    optimizer="AdamW",
    project="ead_classifier",
    name="yolov8m_run1",
    save_period=10,
    val=True,
)
print("Training complete!")
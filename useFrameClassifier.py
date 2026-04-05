import os
import shutil
from ultralytics import YOLO

# ---------------- CONFIG ----------------
INPUT_FOLDER = r"C:\msc\edd2020"
OUTPUT_FOLDER = r"C:\msc\edd2020_classified"

MODEL_PATH = r"runs/classify/train_dgx_2/best.pt"

CLASS_NAMES = {
    0: "both",
    1: "contrast",
    2: "none",
    3: "saturation"
}

# ----------------------------------------

# Load model
model = YOLO(MODEL_PATH)

# Create output folders (exclude "both")
for idx, name in CLASS_NAMES.items():
    if name != "both":
        os.makedirs(os.path.join(OUTPUT_FOLDER, name), exist_ok=True)

# Iterate images
for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)

    # Inference
    results = model(image_path)
    probs = results[0].probs

    # Get full probability vector
    prob_values = probs.data.cpu().numpy()

    # Sort indices by probability (descending)
    sorted_indices = prob_values.argsort()[::-1]

    top1 = sorted_indices[0]
    top2 = sorted_indices[1]

    # Apply rule
    if top1 == 0:  # "both"
        final_class = top2
    else:
        final_class = top1

    class_name = CLASS_NAMES[final_class]

    # Destination path
    dest_folder = os.path.join(OUTPUT_FOLDER, class_name)
    dest_path = os.path.join(dest_folder, filename)

    # Copy file
    shutil.copy(image_path, dest_path)

    print(f"{filename} -> {class_name}")
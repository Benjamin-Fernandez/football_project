import pandas as pd
import os

# --- Configuration ---
INPUT_CSV = "data/raw/gt_test.txt"  # Path to GT file
OUTPUT_DIR = "data/labels/test"  # You can move this to val/test manually
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
NUM_FRAMES = 1802  # From 000001 to 001802

# --- Prepare ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_CSV, header=None)

# --- Generate per-frame YOLO label files ---
for frame_id in range(1, NUM_FRAMES + 1):
    frame_data = df[df[0] == frame_id]
    yolo_lines = []

    for _, row in frame_data.iterrows():
        visibility = row[6]
        class_id = row[7]

        if visibility != 1:
            continue  # Skip invisible objects

        x, y, w, h = row[2], row[3], row[4], row[5]

        x_center = (x + w / 2) / IMAGE_WIDTH
        y_center = (y + h / 2) / IMAGE_HEIGHT
        w_norm = w / IMAGE_WIDTH
        h_norm = h / IMAGE_HEIGHT

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Always write a file, even if empty
    out_path = os.path.join(OUTPUT_DIR, f"{frame_id:06d}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(yolo_lines))

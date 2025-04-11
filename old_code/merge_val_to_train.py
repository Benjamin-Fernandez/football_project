import os
import shutil

# Directories
base_path = "data"
img_val_dir = os.path.join(base_path, "images", "val")
lbl_val_dir = os.path.join(base_path, "labels", "val")
img_train_dir = os.path.join(base_path, "images", "train")
lbl_train_dir = os.path.join(base_path, "labels", "train")

# Starting index for renaming
start_idx = 1803

# Get and sort val image files
val_images = sorted([f for f in os.listdir(img_val_dir) if f.endswith((".jpg", ".png"))])

# Ensure same count exists in labels
val_labels = sorted([f for f in os.listdir(lbl_val_dir) if f.endswith(".txt")])
assert len(val_images) == len(val_labels), "Mismatch between val images and labels."

# Create output folders if they don't exist
os.makedirs(img_train_dir, exist_ok=True)
os.makedirs(lbl_train_dir, exist_ok=True)

# Loop through and move+rename
for i, (img_file, lbl_file) in enumerate(zip(val_images, val_labels)):
    new_id = f"{start_idx + i:06d}"

    # Define new filenames
    new_img_name = new_id + ".jpg"
    new_lbl_name = new_id + ".txt"

    # Full paths
    src_img_path = os.path.join(img_val_dir, img_file)
    src_lbl_path = os.path.join(lbl_val_dir, lbl_file)
    dst_img_path = os.path.join(img_train_dir, new_img_name)
    dst_lbl_path = os.path.join(lbl_train_dir, new_lbl_name)

    # Move and rename
    shutil.move(src_img_path, dst_img_path)
    shutil.move(src_lbl_path, dst_lbl_path)

print(f"Moved and renamed {len(val_images)} images and labels from 'val' to 'train'.")

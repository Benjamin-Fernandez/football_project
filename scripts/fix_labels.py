import os
from tqdm import tqdm

LABELS_DIR = "data/labels/val"  # or data/labels/val

def fix_labels(labels_dir):
    for filename in tqdm(os.listdir(labels_dir)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(labels_dir, filename)

        fixed_lines = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls = int(float(parts[0]))

                # Remap classes
                if cls == 2:
                    parts[0] = "1"  # referee or player → player
                elif cls == 1:
                    parts[0] = "0"  # ball
                else:
                    continue  # skip anything else (e.g. class 0 misused)

                fixed_lines.append(" ".join(parts))

        with open(path, 'w') as f:
            for line in fixed_lines:
                f.write(line + '\n')

# Run for both train and val
fix_labels("data/labels/test")
fix_labels("data/labels/val")

############ THIS IS AN OLD FILE #############

from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/train/football_cpu_fast/weights/best.pt')

# Run prediction on an image
results = model.predict(source='data/images/val/000012.jpg', save=True, conf=0.01)

# OR run prediction on a video
# results = model.predict(source='data/videos/clip.mp4', save=True, conf=0.25)

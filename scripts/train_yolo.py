################# LONGER TRAINING SCRIPT #################
from ultralytics import YOLO

def train():
    model = YOLO('yolov8m.pt')  # Good balance between size and power

    model.train(
        data='data/football.yaml',
        epochs=100,               # Longer training
        imgsz=1280,               # Bigger images help with ball
        batch=8,
        device=0,
        workers=8,
        pretrained=True,
        verbose=True,
        name='football_gpu_train',
        project='runs/train',
        patience=40,              # Let it train longer before stopping
        seed=42,
        deterministic=False,
        save=True,                # Save the best model
        lr0=0.001,                # You can tune learning rate here
        warmup_epochs=3,
        hsv_h=0.015,              # Light color augmentation
        hsv_s=0.5,
        hsv_v=0.5,
        mosaic=0.5,               # Keep mosaic limited for small objects
        mixup=0.0                 # Disable mixup (not great for small objects)
    )

    return model

if __name__ == "__main__":
    model = train()







################## OLD GPU TRAINING SCRIPT ##################

# from ultralytics import YOLO

# def train():
#     model = YOLO('yolov8m.pt')  # Medium model: balanced accuracy & speed

#     model.train(
#         data='data/football.yaml',
#         epochs=50,              # More epochs since you now have GPU power
#         imgsz=960,              # Higher resolution for better detection
#         batch=16,               # Increase batch size (GPU has 24GB VRAM)
#         device=0,               # Use first GPU (GPU:0)
#         workers=8,              # Use more CPU workers to load data
#         pretrained=True,
#         verbose=True,
#         name='football_gpu_train',
#         project='runs/train',
#         patience=10,
#         seed=42,
#         deterministic=False,    # Let GPU optimize execution
#         save_period=5           # Save every 5 epochs
#     )

#     return model

# if __name__ == "__main__":
#     model = train()




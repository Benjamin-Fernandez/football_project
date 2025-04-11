######################################### BOXES #########################################


from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Define colors
CLASS_COLORS = {
    0: (0, 0, 255),   # Ball - Red
    1: (0, 255, 0),   # Player - Green
}

def track_and_visualize(model_path, image_folder, save_dir):
    model = YOLO(model_path)

    image_paths = sorted(Path(image_folder).glob("*.jpg"))
    os.makedirs(save_dir, exist_ok=True)

    tracker_stream = model.track(
        source=image_folder,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.1,
        iou=0.45
    )

    for frame_path, result in zip(image_paths, tracker_stream):
        im0 = result.orig_img.copy()

        if result.boxes.cls is not None:
            for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
                cls_id = int(cls_id)

                if cls_id not in [0, 1]:
                    continue

                x1, y1, x2, y2 = map(int, box)
                color = CLASS_COLORS[cls_id]
                # label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

                thickness = 1 if cls_id == 0 else 2  # thinner for ball
                cv2.rectangle(im0, (x1, y1), (x2, y2), color, thickness)
                # cv2.putText(im0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        save_path = os.path.join(save_dir, frame_path.name)
        cv2.imwrite(save_path, im0)
        print(f"Processed {frame_path.name}")

if __name__ == "__main__":
    track_and_visualize(
        model_path="runs/train/football_gpu_train2/weights/best.pt",
        image_folder="data/images/test",
        save_dir="runs/track/test"
    )







######################### CIRCLES ############################


# from ultralytics import YOLO
# import cv2
# import os
# from pathlib import Path

# # New color and size settings
# CLASS_COLORS = {
#     0: (0, 0, 255),   # Ball - red
#     1: (0, 255, 0),   # Player - green
# }
# CLASS_RADIUS = {
#     0: 3,  # Ball
#     1: 6   # Player
# }

# def track_and_visualize(model_path, image_folder, save_dir):
#     model = YOLO(model_path)

#     # Get sorted list of .jpg images
#     image_paths = sorted(Path(image_folder).glob("*.jpg"))
#     os.makedirs(save_dir, exist_ok=True)

#     # Run tracker
#     tracker_stream = model.track(
#         source=image_folder,
#         stream=True,
#         persist=True,
#         tracker="bytetrack.yaml",
#         conf=0.1,
#         iou=0.45
#     )

#     for frame_path, result in zip(image_paths, tracker_stream):
#         im0 = result.orig_img.copy()

#         if result.boxes.cls is not None:
#             for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
#                 cls_id = int(cls_id)

#                 # Only keep ball (0) and player (1)
#                 if cls_id not in [0, 1]:
#                     continue

#                 x1, y1, x2, y2 = map(int, box)
#                 cx = int((x1 + x2) / 2)
#                 cy = int((y1 + y2) / 2)
#                 color = CLASS_COLORS[cls_id]
#                 radius = CLASS_RADIUS[cls_id]

#                 # Draw a solid circle
#                 cv2.circle(im0, (cx, cy), radius, color, -1)

#         save_path = os.path.join(save_dir, frame_path.name)
#         cv2.imwrite(save_path, im0)
#         print(f"Processed {frame_path.name}")

# if __name__ == "__main__":
#     track_and_visualize(
#         model_path="runs/train/football_gpu_train6/weights/best.pt",
#         image_folder="data/images/val",
#         save_dir="runs/track/val"
#     )







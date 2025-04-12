
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


    # Tracker params override for better ID stability
    tracker_stream = model.track(
        source=image_folder,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.1,
        iou=0.5
    )

    for frame_path, result in zip(image_paths, tracker_stream):
        im0 = result.orig_img.copy()

        if result.boxes.cls is not None:
            for box, cls_id, track_id in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.id):
                cls_id = int(cls_id)
                if cls_id not in [0, 1]:
                    continue

                x1, y1, x2, y2 = map(int, box)
                color = CLASS_COLORS[cls_id]
                thickness = 1 if cls_id == 0 else 2

                cv2.rectangle(im0, (x1, y1), (x2, y2), color, thickness)

                # Show ID only for players
                if track_id is not None:
                    label = f"ID {int(track_id)}"
                    cv2.putText(im0, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        save_path = os.path.join(save_dir, frame_path.name)
        cv2.imwrite(save_path, im0)
        print(f"Processed {frame_path.name}")

if __name__ == "__main__":
    track_and_visualize(
        model_path="runs/train/football_gpu_train2/weights/best.pt",
        image_folder="data/images/test",
        save_dir="runs/track/test_id"
    )   



# ######################################### BOXES #########################################


# from ultralytics import YOLO
# import cv2
# import os
# from pathlib import Path

# # Define colors
# CLASS_COLORS = {
#     0: (0, 0, 255),   # Ball - Red
#     1: (0, 255, 0),   # Player - Green
# }

# def track_and_visualize(model_path, image_folder, save_dir):
#     model = YOLO(model_path)

#     image_paths = sorted(Path(image_folder).glob("*.jpg"))
#     os.makedirs(save_dir, exist_ok=True)

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

#                 if cls_id not in [0, 1]:
#                     continue

#                 x1, y1, x2, y2 = map(int, box)
#                 color = CLASS_COLORS[cls_id]
#                 # label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

#                 thickness = 1 if cls_id == 0 else 2  # thinner for ball
#                 cv2.rectangle(im0, (x1, y1), (x2, y2), color, thickness)
#                 # cv2.putText(im0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

#         save_path = os.path.join(save_dir, frame_path.name)
#         cv2.imwrite(save_path, im0)
#         print(f"Processed {frame_path.name}")

# if __name__ == "__main__":
#     track_and_visualize(
#         model_path="runs/train/football_gpu_train2/weights/best.pt",
#         image_folder="data/images/test",
#         save_dir="runs/track/test"
#     )




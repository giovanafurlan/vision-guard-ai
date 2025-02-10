from ultralytics import YOLO
import cv2
import os

# Load a pretrained YOLOv8 model
model = YOLO("yolov8x.pt")  # 'yolov8x.pt' is the most powerful version

# Path to images
image_folder = "/path/to/images"  # Change this to your folder
output_folder = "/path/to/annotations"  # Where YOLO .txt files will be saved
os.makedirs(output_folder, exist_ok=True)

# Process each image
for image_name in os.listdir(image_folder):
    if not image_name.endswith((".jpg", ".png", ".jpeg")):
        continue  # Skip non-image files

    image_path = os.path.join(image_folder, image_name)
    img = cv2.imread(image_path)

    # Run YOLO object detection
    results = model(image_path)

    # Save annotation file
    txt_filename = os.path.join(output_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
    with open(txt_filename, "w") as f:
        for r in results:
            for box in r.boxes.xywhn:  # Normalized YOLO format (x_center, y_center, width, height)
                x_center, y_center, width, height = box[:4]  # Get bounding box coordinates
                class_id = int(r.boxes.cls[0])  # Detected class ID
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("✅ Annotations saved in YOLO format!")

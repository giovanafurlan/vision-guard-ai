import os
import cv2
import tempfile
from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "runs", "detect", "train", "weights", "best.pt")

def process_video(video_path, output_dir, skip_frames=5):
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    os.makedirs(output_dir, exist_ok=True)
    prints_dir = os.path.join(output_dir, "prints")
    os.makedirs(prints_dir, exist_ok=True)

    frame_count = 0
    print_count = 0
    found_objects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            results = model(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    confidence = box.conf.item()
                    if confidence > 0.5:
                        cropped_object = frame[y1:y2, x1:x2]
                        if cropped_object.size > 0:
                            print_file = os.path.join(prints_dir, f"print_{print_count}.jpg")
                            cv2.imwrite(print_file, cropped_object)
                            found_objects.append(print_file)
                            print_count += 1
        frame_count += 1

    cap.release()
    return found_objects

@app.route("/process-video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "input.mp4")
    video_file.save(video_path)

    output_dir = os.path.join(temp_dir, "output")
    processed_images = process_video(video_path, output_dir)

    return jsonify({"message": "Processing complete", "images": processed_images})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import tempfile
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "runs", "detect", "train", "weights", "best.pt")

UPLOAD_FOLDER = os.path.join(BASE_PATH, "static", "images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_video(video_path, output_dir, skip_frames=5):
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    os.makedirs(output_dir, exist_ok=True)
    found_objects = []

    frame_count = 0
    print_count = 0

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
                            filename = f"print_{print_count}.jpg"
                            file_path = os.path.join(UPLOAD_FOLDER, filename)
                            cv2.imwrite(file_path, cropped_object)
                            found_objects.append(f"http://127.0.0.1:5001/static/images/{filename}")
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

    processed_images = process_video(video_path, UPLOAD_FOLDER)

    return jsonify({"message": "Processing complete", "images": processed_images})

@app.route("/static/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

import cv2
from ultralytics import YOLO
import os

# Definir caminho base para facilitar mudanças futuras
BASE_PATH = r"C:\git\Postech\Hackaton\vision-guard-ai-main"

def process_video_with_prints(video_path, output_path, model_path=None, skip_frames=5):
    # Definir o caminho do modelo, se não for passado
    if model_path is None:
        model_path = os.path.join(BASE_PATH, "runs", "detect", "train", "weights", "best.pt")

    # Load the trained YOLO model
    model = YOLO(model_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create directory for individual frame prints
    prints_dir = os.path.join(output_path, "prints")
    os.makedirs(prints_dir, exist_ok=True)

    # Create video writer for saving the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_path, "processed_video.mp4"), fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    print_count = 0  # Count of saved prints

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to speed up processing
        if frame_count % skip_frames == 0:
            # Run YOLO inference
            results = model(frame)

            # Draw bounding boxes and save individual prints
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Bounding box coordinates
                    confidence = box.conf.item()  # Confidence score
                    if confidence > 0.5:  # Confidence threshold
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame, f"Sharp Object {confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
                        
                        # Save the detected region as an individual print
                        cropped_object = frame[y1:y2, x1:x2]
                        if cropped_object.size > 0:  # Ensure valid region
                            print_file = os.path.join(prints_dir, f"print_{print_count}.jpg")
                            cv2.imwrite(print_file, cropped_object)
                            print_count += 1

        # Write processed frame to the output video
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Video processing complete. Processed video saved to {output_path}/processed_video.mp4")
    print(f"Prints saved to {prints_dir}")

if __name__ == "__main__":
    # Caminhos ajustados para o novo ambiente
    video_path = os.path.join(BASE_PATH, "input_video.mp4")  # Substitua pelo nome real do vídeo
    output_path = os.path.join(BASE_PATH, "output")
    model_path = os.path.join(BASE_PATH, "runs", "detect", "train", "weights", "best.pt")

    # Criar diretório de saída, se necessário
    os.makedirs(output_path, exist_ok=True)

    # Executar a função
    process_video_with_prints(
        video_path=video_path,
        output_path=output_path,
        model_path=model_path,
        skip_frames=5  # Ajuste conforme necessário
    )

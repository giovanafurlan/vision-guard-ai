from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('/Users/giovanafurlan/Documents/GitHub/vision-guard-ai/runs/detect/train/weights/best.pt')

# Load a test image
img = cv2.imread('knife_24.jpg')

# Run inference
results = model(img)

# Display the results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        confidence = box.conf.item()
        print(f"Bounding Box: {x1}, {y1}, {x2}, {y2}, Confidence: {confidence:.2f}")

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, f"Sharp Object {confidence:.2f}",
            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

# Show the image with detections
cv2.imshow('Detections', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

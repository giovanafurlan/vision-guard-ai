from ultralytics import YOLO

# Train YOLOv5 model with enhancements
def train_model():
    # Load pretrained YOLOv5s model
    model = YOLO('yolov5s.pt')

    # Visualize dataset before training
    print("Validating dataset structure...")
    model.val(data='/Users/giovanafurlan/Documents/GitHub/vision-guard-ai/sharp_objects_dataset/sharp_objects_dataset.yaml', plots=True)

    # Start training with enhanced configuration
    model.train(
        data='/Users/giovanafurlan/Documents/GitHub/vision-guard-ai/sharp_objects_dataset/sharp_objects_dataset.yaml',  # Path to YAML file
        epochs=40,  # Increase number of epochs for better convergence
        imgsz=640,  # Image size
        batch=16,  # Batch size
        freeze=[0],  # Freeze backbone layers to focus on detection layers
        close_mosaic=10,  # Stop mosaic augmentation after 10 epochs
        lr0=0.001,  # Lower learning rate to prevent overshooting
    )

    # Save the best model
    print("Training completed. Saving the model...")
    model.export(format="torchscript")  # Export trained model in TorchScript format

    return model


# Run training
if __name__ == "__main__":
    trained_model = train_model()

    # Test the trained model on an example image
    print("Testing the model on a sample image...")
    results = trained_model('/Users/giovanafurlan/Documents/GitHub/vision-guard-ai/sharp_objects_dataset/images/test/sample_image.jpg', conf=0.1)
    results.save()  # Save prediction results
    print("Predictions saved!")

from ultralytics import YOLO

# Load model
model = YOLO("yolo26n.pt")

# Train
model.train(
    data="cfg/CV-LP-ocr.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device="mps",  # '0' para GPU, 'cpu' para CPU, 'mps' para Mac
    name="ocr_detector",
)

print("\n✅ Training complete!")

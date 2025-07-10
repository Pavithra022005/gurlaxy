# detect_and_control.py

from ultralytics import YOLO
import cv2
import serial
import time

# ========== CONFIGURATION ==========
YOLO_MODEL = "yolov8n.pt"          # Pretrained YOLOv8 model
TARGET_OBJECT = "toothbrush"       # Change to "spoon", "cup", etc.
SERIAL_PORT = "COM3"               # Change this for your laptop (Linux: '/dev/ttyUSB0' or '/dev/ttyACM0')
BAUD_RATE = 9600
CONFIDENCE_THRESHOLD = 0.6
# ===================================

# Initialize Serial (comment out if Arduino not connected)
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # wait for serial to initialize
    print(f"Connected to Arduino on {SERIAL_PORT}")
except:
    arduino = None
    print("Arduino not connected. Running in test mode.")

# Load YOLOv8 model
model = YOLO(YOLO_MODEL)

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for laptop webcam

print("Starting object detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Run YOLO object detection
    results = model(frame, verbose=False)
    detections = results[0].boxes

    for box in detections:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw box (for visualization)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # If detected target object
        if label.lower() == TARGET_OBJECT.lower() and conf > CONFIDENCE_THRESHOLD:
            print(f"{TARGET_OBJECT} detected with confidence {conf:.2f}")
            if arduino:
                arduino.write(b"GRAB\n")
                print("Command 'GRAB' sent to Arduino")
            time.sleep(2)  # Wait before next detection to prevent spamming

    cv2.imshow("Smart Glove Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

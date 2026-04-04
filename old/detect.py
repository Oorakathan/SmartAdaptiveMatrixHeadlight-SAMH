import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the model - using 'track' later is better for Re-ID
model = YOLO("yolov8n.pt")

stream_url = "http://192.168.0.5:81/stream"
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

# Balanced Sharpening Kernel (less aggressive to avoid artifacts)
sharpen_kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])

print("Optimizing for 480x320 resolution...")

while True:
    ret, frame = cap.read()
    if not ret: continue

    # --- QUALITY ENHANCEMENT PIPELINE ---
    
    # 1. Denoising: ESP32-CAM at low res has 'salt and pepper' noise.
    # Median blur helps remove artifacts without blurring edges like Gaussian would.
    denoised = cv2.medianBlur(frame, 3)

    # 2. Contrast & Sharpening
    yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    final_input = cv2.filter2D(enhanced, -1, sharpen_kernel)

    # --- ACCURACY-FOCUSED INFERENCE ---
    
    # We use model.track() instead of model() for your Re-ID project.
    # imgsz=480 tells YOLO to scale based on your longest dimension.
    # persist=True maintains IDs even if a player is briefly obscured.
    results = model.track(final_input, 
                          persist=True, 
                          classes=[0], 
                          conf=0.3, 
                          imgsz=480, 
                          tracker="bytetrack.yaml") # Bytetrack is better for low-res

    for r in results:
        # Drawing bounding boxes and IDs
        annotated_frame = r.plot()
        
        # Accessing IDs for your Re-ID logic
        if r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            # print(f"Active Player IDs: {ids}")

    cv2.imshow("Enhanced 480x320 Feed", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
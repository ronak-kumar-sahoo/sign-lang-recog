import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys

# Load model with safety check
try:
    model = load_model('sign_model_kaggle.h5')
    classes = 'ABCDEFGHIKLMNOPQRSTUVWXY'  # 25 letters exactly
    print("✓ Model loaded!")
except:
    print("❌ Run: python train_model.py first!")
    input("Press Enter to exit...")
    sys.exit()

# Auto-detect working camera
cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✓ Camera {i} working!")
        break
    cap.release()

if not cap or not cap.isOpened():
    print("❌ No webcam found! Close Zoom/Teams")
    input("Press Enter to exit...")
    sys.exit()

roi_top, roi_left, roi_w, roi_h = 150, 200, 250, 250

print("\n🎬 ASL RECOGNITION LIVE")
print("📱 Put RIGHT HAND in GREEN BOX")
print("⏹️  Press Q to QUIT")
print("-" * 40)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Camera lost, retrying...")
        continue
        
    # Extract hand region
    roi = frame[roi_top:roi_top+roi_h, roi_left:roi_left+roi_w]
    if roi.size == 0:
        continue
        
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    gray_normalized = gray_resized.astype(np.float32) / 255.0
    
    # SAFE prediction
    try:
        pred = model.predict(gray_normalized.reshape(1,28,28,1), verbose=0)
        if pred.shape[1] == 25:  # Correct shape check
            letter_idx = np.argmax(pred[0])
            letter = classes[letter_idx]
            confidence = pred[0][letter_idx] * 100
        else:
            letter, confidence = "?", 0
    except:
        letter, confidence = "ERROR", 0
    
    # Draw results
    cv2.rectangle(frame, (roi_left, roi_top), (roi_left+roi_w, roi_top+roi_h), (0,255,0), 3)
    cv2.putText(frame, f"{letter} ({confidence:.0f}%)", (roi_left+10, roi_top-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    # Instructions
    cv2.putText(frame, "Q=QUIT", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, "Hand in GREEN BOX", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow('ASL RECOGNITION - Press Q', frame)
    
    # 30fps stable loop
    key = cv2.waitKey(33) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Demo closed!")

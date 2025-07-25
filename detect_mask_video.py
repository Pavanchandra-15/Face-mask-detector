import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Load pre-trained model
model = load_model("mask_detector.h5")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("face_detector/haarcascade_frontalface_default.xml")


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (224, 224))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 224, 224, 3))
        
        result = model.predict(reshaped)
        label = "Mask" if result[0][0] > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Mask Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

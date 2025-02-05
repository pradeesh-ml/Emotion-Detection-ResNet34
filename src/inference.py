import cv2
import numpy as np
from keras.models import load_model

# Load trained model
model = load_model("models/model.h5")
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load Haarcascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_region = cv2.resize(face_region, (48, 48))
        face_region = np.reshape(face_region, (1, 48, 48, 1)) / 255.0
        
        prediction = model.predict(face_region)
        emotion = emotions[np.argmax(prediction)]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

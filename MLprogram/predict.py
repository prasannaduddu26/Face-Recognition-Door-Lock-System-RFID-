import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.svm import SVC
import joblib
from mtcnn import MTCNN
import serial
import time

# Serial Communication
ard = serial.Serial(port='COM3', baudrate=115200, timeout=1)
time.sleep(2)

def predict_face():
    model = joblib.load('facenet_svm_model.pkl')
    facenet = FaceNet()
    detector = MTCNN()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    last_label = None
    last_sent_time = 0
    cooldown_seconds = 5  # Time to allow re-send if face disappears

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        current_time = time.time()
        detected_label = None

        for face in faces:
            x, y, w, h = face['box']
            face_img = rgb_frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))

            embedding = facenet.embeddings([face_img])[0]
            pred_label = model.predict([embedding])[0]
            prob = model.predict_proba([embedding])[0]
            confidence = np.max(prob)

            label = pred_label if confidence > 0.7 else 'Unknown'
            detected_label = label

            # Only send command if label is new or timeout passed
            if label in ["prasanna", "harika"]:
                if (label != last_label) or (current_time - last_sent_time > cooldown_seconds):
                    command = 'A' if label == "prasanna" else 'B'
                    ard.write(command.encode())
                    print(f"Sent '{command}' for {label}")
                    last_label = label
                    last_sent_time = current_time

            # Draw on frame
            color = (0, 255, 0) if label != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{label} ({confidence*100:.2f}%)', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # If no face detected, reset label after cooldown
        if not faces and (current_time - last_sent_time > cooldown_seconds):
            last_label = None

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    ard.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_face()

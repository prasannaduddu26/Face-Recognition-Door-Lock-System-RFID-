import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.svm import SVC
import joblib
from mtcnn import MTCNN
import serial
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time


# ---------- Serial Setup ----------
ser = serial.Serial('COM3', 9600)  # Change COM3 to your serial port
time.sleep(2)

# ---------- Email Setup ----------
def send_email_alert():
    sender_email = 'aswiniprathipati02@gmail.com'
    sender_password = 'pddu bxmn wbqp rcek'  # Use app password, not your main password
    receiver_email = 'prasanna4219@gmail.com'

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'Unknown Face Detected'
    msg.attach(MIMEText('An unknown person was detected by the system.', 'plain'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("Email sent for unknown face.")
    except Exception as e:
        print("Failed to send email:", str(e))

# ---------- Main Prediction Function ----------
def predict_face():
    model = joblib.load('facenet_svm_model.pkl')
    facenet = FaceNet()
    detector = MTCNN()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    unknown_sent = False  # Flag to prevent email spamming

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face['box']
            face_img = rgb_frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))

            embedding = facenet.embeddings([face_img])[0]

            pred_label = model.predict([embedding])[0]
            prob = model.predict_proba([embedding])[0]
            confidence = np.max(prob)

            label = pred_label if confidence > 0.7 else 'Unknown'
            color = (0, 255, 0) if label != 'Unknown' else (0, 0, 255)

            # ---------- Serial and Email Logic ----------
            if label == 'Unknown':
                if not unknown_sent:
                    send_email_alert()
                    unknown_sent = True
            else:
                unknown_sent = False
                if label.lower() == 'aswini':  # Match your specific known person
                    ser.write('A'.encode())
                
                if label.lower() == 'pavani':  # Match your specific known person
                    ser.write('B'.encode())
                


            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{label} ({confidence*100:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_face()

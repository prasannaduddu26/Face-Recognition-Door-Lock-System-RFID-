import os
import numpy as np
import cv2
from keras_facenet import FaceNet
from sklearn.svm import SVC
import joblib

def load_dataset(dataset_path='dataset'):
    facenet = FaceNet()
    X, y = [], []

    for user_name in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_name)
        if not os.path.isdir(user_path):
            continue

        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (160, 160))
            
            # Generate face embedding
            embedding = facenet.embeddings([img])[0]
            X.append(embedding)
            y.append(user_name)

    X = np.array(X)
    y = np.array(y)
    return X, y

def train_model():
    print("Loading dataset...")
    X, y = load_dataset()

    print("Training model...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)

    # Save the model
    joblib.dump(model, 'facenet_svm_model.pkl')
    print("Model trained and saved as 'facenet_svm_model.pkl'")

if __name__ == "__main__":
    train_model()

import cv2
import os

def create_dataset(username, num_images=100):
    # Create a directory to store images
    dataset_path = 'dataset'
    user_path = os.path.join(dataset_path, username)
    os.makedirs(user_path, exist_ok=True)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print(f"Collecting {num_images} images for {username}. Look at the camera...")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow('Collecting Images', frame)

        # Save the frame
        img_name = os.path.join(user_path, f'{username}_{count}.jpg')
        cv2.imwrite(img_name, frame)
        count += 1

        # Break when enough images are collected
        if count >= num_images:
            print(f"Collected {num_images} images for {username}.")
            break

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    username = input("Enter the username: ").strip()
    create_dataset(username)

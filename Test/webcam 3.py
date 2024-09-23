import cv2
import os
import re
import pickle
import face_recognition
import numpy as np
from EncodeGenrator import EG
from datetime import datetime
from db_connection import connect_to_database, create_table, insert_image_data, close_connection

def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height
    imgBackground = cv2.imread("Modes/background.png")

    if not cap.isOpened():
        print("ERROR: Could not open the camera")
        return None, None

    return cap, imgBackground

def load_encodings():
    encodeListKnown, empNames = [], []
    if os.path.exists('EncodeFile.p'):
        print("Loading The Encoding File.....")
        with open('EncodeFile.p', 'rb') as file:
            encodeListKnownWithNames = pickle.load(file)
            encodeListKnown, empNames = encodeListKnownWithNames
        print("Encoding File Loaded....")
    else:
        print("EncodeFile.p not found. Starting with an empty encoding list.")
    return encodeListKnown, empNames

def save_image(employee_name, frame, image_counter):
    image_path = f'Employee_Images/{employee_name}_{image_counter}.jpg'
    cv2.imwrite(image_path, frame)
    print(f"Image saved as {image_path}")
    return image_path

def prompt_for_name():
    return input("Please enter your name in FirstName_LastName format:")

def capture_image():
    # Initialize camera
    cap, imgBackground = initialize_camera()
    if cap is None:
        return

    # Create Employee Images directory if it doesn't exist
    os.makedirs('Employee_Images', exist_ok=True)

    # Load existing encodings
    encodeListKnown, empNames = load_encodings()

    # Connect to the database and create table if it doesn't exist
    connection = connect_to_database()
    if connection:
        create_table(connection)

    image_counter = 1
    employee_name = None  # Initialize employee_name as None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        imgBackground[162:162+480,55:55+640] = frame

        # Resize and convert frame for face recognition
        frameS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

        faceCurrFrame = face_recognition.face_locations(frameS)
        encodeCurrFrame = face_recognition.face_encodings(frameS, faceCurrFrame)

        face_detected = False
        for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            if faceDis.size > 0:
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    face_detected = True
                    print("Known Face Detected.")
                    capture_date = datetime.now().strftime("%d-%m-%y")
                    capture_time = datetime.now().strftime("%I:%M:%S%p")
                    print(f"Detected: {empNames[matchIndex]} at {capture_time} on {capture_date}")

                    realtime_image_path = f'Employee_Images/{empNames[matchIndex]}.jpg'
                    cv2.imwrite(realtime_image_path, frame)

                    insert_image_data(connection, empNames[matchIndex], realtime_image_path, face_detected)

        if not face_detected and not encodeListKnown:
            if employee_name is None:
                # Prompt for name only if no known faces are detected and no encodings exist
                employee_name = prompt_for_name()

            print("New face detected. Please press 'c' to capture an image or 'q' for exiting the program.")

            # Show the frame to capture image for the new employee
            cv2.imshow('Webcam', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                existing_files = os.listdir('Employee_Images')
                pattern = re.compile(rf'{employee_name}_(\d+)\.jpg')
                for filename in existing_files:
                    match = pattern.match(filename)
                    if match:
                        number = int(match.group(1))
                        if number >= image_counter:
                            image_counter = number + 1

                image_path = save_image(employee_name, frame, image_counter)
                image_counter += 1
                EG(image_path)
                encodeListKnown, empNames = load_encodings()

                insert_image_data(connection, employee_name, 'Employee_Images', face_detected)
                continue

        # Display the captured frame
        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting the program.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if connection:
        close_connection(connection)

if __name__ == "__main__":
    capture_image()

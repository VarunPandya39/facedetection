import cv2
import os
import re
import pickle
import face_recognition
import numpy as np
from EncodeGenrator import EG
from datetime import datetime
from db_connection import connect_to_database, create_table, insert_image_data, close_connection

def capture_image(employee_name):

    #Create Employee Images directory if it doesn't exist
    if not os.path.exists('Employee_Images'):
        os.makedirs('Employee_Images')

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    imgBackground = cv2.imread("Modes/background.png")

    if not cap.isOpened():
        print("ERROR: Could not open the camera")
        return

    # Check if Employee_Images folder is empty or EncodeFile.p is missing
    encoding_needed = not os.path.exists('EncodeFile.p') or not os.listdir('Employee_Images')

    if encoding_needed:
        print("Employee_Images folder is empty or EncodeFile.p is missing.")
        #print("Press 'c' to capture an image, or 'q' to quit.")

    # Print statement before capturing
    #print("Press 'c' to capture an image, or 'q' to quit.")

    image_counter = 1
    existing_files = os.listdir('Employee_Images')
    pattern = re.compile(rf'{employee_name}_(\d+)\.jpg')

    for filename in existing_files:
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number >= image_counter:
                image_counter = number + 1

    # #Load the encoding file
    # print("Loading The Encoding File.....")
    # file = open('EncodeFile.p','rb')
    # encodeListKnownWithNames = pickle.load(file)
    # file.close()
    # encodeListKnown,empNames = encodeListKnownWithNames
    # print(empNames)
    # print("Encoding File Loaded....")

    # Load the encoding file if it exists
    encodeListKnown, empNames = [], []
    if os.path.exists('EncodeFile.p'):
        print("Loading The Encoding File.....")
        with open('EncodeFile.p', 'rb') as file:
            encodeListKnownWithNames = pickle.load(file)
            encodeListKnown, empNames = encodeListKnownWithNames
        print("Encoding File Loaded....")
    else:
        print("EncodeFile.p not found. Starting with an empty encoding list.")

    # Connect to the database and create table if it doesn't exist
    connection = connect_to_database()
    if connection:
        create_table(connection)

    message_printed = False
    while cap.isOpened():
        ret, frame = cap.read()

        imgBackground[162:162+480,55:55+640] = frame

        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        if encoding_needed:
            cv2.imshow('Webcam', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):

                image_path = f'Employee_Images/{employee_name}_{image_counter}.jpg'
                cv2.imwrite(image_path, frame)
                print(f"Image saved as {image_path}")
                image_counter += 1

                # Generate encodings and save the encoding file
                EG(image_path)
                encoding_needed = False

                # Reload the encoding file
                if os.path.exists('EncodeFile.p'):
                    print("Reloading The Encoding File.....")
                    with open('EncodeFile.p', 'rb') as file:
                        encodeListKnownWithNames = pickle.load(file)
                        encodeListKnown, empNames = encodeListKnownWithNames
                    print("Encoding File Reloaded....")
                else:
                    print("Failed to generate EncodeFile.p")
                continue

        #Resizing the image so it doesnt take alot of computational power
        frameS = cv2.resize(frame,(0,0),None,0.25,0.25)
        frameS = cv2.cvtColor(frameS,cv2.COLOR_BGR2RGB)

        #Feeding the value to the face recognition system so it detects and gives us an output
        #so we feed in the faces and encodings in the current frame
        #we compare the faceEncodings of the new faces in the current frame and compare it with the previous encodings
        faceCurrFrame = face_recognition.face_locations(frameS)
        encodeCurrFrame = face_recognition.face_encodings(frameS,faceCurrFrame)

        face_detected = False

        #Looping through the currFrame encodings and then comparing it with the genrated encodings
        for encodeFace,faceLoc in zip(encodeCurrFrame,faceCurrFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print("Matches", matches)
            #print("FaceDis", faceDis)
            if faceDis.size > 0:  # Check if faceDis is not empty
                matchIndex = np.argmin(faceDis)

            #print(matchIndex)
            # y1,x2,y2,x1 = faceLoc #Coordinates for face locations
            # y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 #Multiplying by 4 because we reduced the size of img by 1/4
            # frame = cv2.rectangle(frame, (y1, x2), (y2, x1), (255,0,0), 2)

            # if matchIndex < len(matches) and matches[matchIndex]:
                if matches[matchIndex]:
                    face_detected = True

                    print("Known Face Detected.")
                    #print(empNames[matchIndex])
                    capture_date = datetime.now().strftime("%d-%m-%y")
                    capture_time = datetime.now().strftime("%I:%M:%S%p")
                    print(f"Detected: {empNames[matchIndex]} at {capture_time} on {capture_date}")

                    # Save the real-time image when the face is detected
                    #realtime_image_path = f'Employee_Images/{empNames[matchIndex]}_{capture_date}_{capture_time}.jpg'
                    realtime_image_path = f'Employee_Images/{empNames[matchIndex]}.jpg'
                    cv2.imwrite(realtime_image_path, frame)  # Save the current frame as an image

                    # Insert the detection event into the database
                    #insert_image_data(connection, empNames[matchIndex], 'Employee_Images', face_detected)

                    insert_image_data(connection, empNames[matchIndex], realtime_image_path, face_detected)

        if not face_detected:
            cv2.imshow('Webcam', frame)
            key = cv2.waitKey(1) & 0xFF

            if not message_printed:
                print("New Employee Detected")
                print("Press 'C' to capture an image for genrating face encodings.")
                print("Press 'Q' for exiting the program.")
                message_printed = True

            if key == ord('c'):
                image_path = f'Employee_Images/{employee_name}_{image_counter}.jpg'
                cv2.imwrite(image_path, frame)
                print(f"Image saved as {image_path}")
                image_counter += 1

                # Generate encodings and save the encoding file
                EG(image_path)
                # Reload the encoding file
                if os.path.exists('EncodeFile.p'):
                    print("Reloading The Encoding File.....")
                    with open('EncodeFile.p', 'rb') as file:
                        encodeListKnownWithNames = pickle.load(file)
                        encodeListKnown, empNames = encodeListKnownWithNames
                    print("Encoding File Reloaded....")
                else:
                    print("Failed to generate EncodeFile.p")
                continue

        # Display the captured frame
        cv2.imshow('Webcam', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exiting the program.")
            break
        elif key == ord('c'):
            #timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            #unique_id = uuid.uuid4().hex  # Generates a random UUID

            #Capture an image
            image_path = f'Employee_Images/{employee_name}_{image_counter}.jpg'
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            image_counter += 1

            # EG(image_path,encodeListKnown,empNames)
            EG(image_path)

            # Insert the captured image into the database
            insert_image_data(connection, employee_name, 'Employee_Images', face_detected)

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    #break

    # Close the database connection
    if connection:
        close_connection(connection)


# Input employee's name
employee_name = input("Please enter your name in FirstName_LastName format:")
capture_image(employee_name)


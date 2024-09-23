import cv2
import os
import re
import pickle
import face_recognition
import numpy as np
from EncodeGenrator import EG
from datetime import datetime
from db_connection import connect_to_database, create_table, insert_image_data, close_connection
def capture_image():
    # Create Employee Images directory if it doesn't exist
    if not os.path.exists('Employee_Images'):
        os.makedirs('Employee_Images')

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    if not cap.isOpened():
        print("ERROR: Could not open the camera")
        return

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

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        # Resize and process frame
        frameS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

        # Detect faces and encodings
        faceCurrFrame = face_recognition.face_locations(frameS)
        encodeCurrFrame = face_recognition.face_encodings(frameS, faceCurrFrame)

        face_detected = False
        for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            if matches and any(matches):
                if faceDis.size > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        face_detected = True
                        print("Known Face Detected.")
                        capture_date = datetime.now().strftime("%d-%m-%y")
                        capture_time = datetime.now().strftime("%I:%M:%S%p")
                        print(f"Detected: {empNames[matchIndex]} at {capture_time} on {capture_date}")

                        # Save the real-time image when the face is detected
                        realtime_image_path = f'Employee_Images/{empNames[matchIndex]}.jpg'
                        cv2.imwrite(realtime_image_path, frame)

                        # Insert the detection event into the database
                        insert_image_data(connection, empNames[matchIndex], realtime_image_path, face_detected)
                        break
            else:
                print("No matches found.")
                print("No known face detected.")

                # Prompt for the user's name
                employee_name = input("New person detected. Please enter your name in FirstName_LastName format: ")

                while True:
                    # Display the frame
                    cv2.imshow('Webcam', frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('c'):
                        # Initialize image_counter and get existing files
                        image_counter = 1
                        existing_files = os.listdir('Employee_Images')
                        pattern = re.compile(rf'{employee_name}_(\d+)\.jpg')

                        for filename in existing_files:
                            match = pattern.match(filename)
                            if match:
                                number = int(match.group(1))
                                if number >= image_counter:
                                    image_counter = number + 1

                        image_path = f'Employee_Images/{employee_name}_{image_counter}.jpg'
                        cv2.imwrite(image_path, frame)
                        print(f"Image saved as {image_path}")

                        try:
                            # Generate encodings and save the encoding file
                            EG(image_path)
                            if os.path.exists('EncodeFile.p'):
                                print("Reloading The Encoding File.....")
                                with open('EncodeFile.p', 'rb') as file:
                                    encodeListKnownWithNames = pickle.load(file)
                                    encodeListKnown, empNames = encodeListKnownWithNames
                                print("Encoding File Reloaded....")
                            else:
                                print("Failed to generate EncodeFile.p")

                            # Insert the captured image into the database
                            insert_image_data(connection, employee_name, image_path, False)
                        except Exception as e:
                            print(f"An error occurred: {e}")
                        break

                    elif key == ord('q'):
                        print("Exiting the program.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # Once a new image is captured and processed, redirect back to capturing frames from the camera
                print("Returning to capturing frames...")
                break

        #Always display the webcam feed
        cv2.imshow('Webcam', frame)

        #Exit thw loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Close the database connection
    if connection:
        close_connection(connection)

# Start the process
capture_image()




# import cv2
# import os
# import re
# import pickle
# import face_recognition
# import numpy as np
# import threading
# import queue
# from EncodeGenrator import EG
# from datetime import datetime
# from db_connection import connect_to_database, create_table, insert_image_data, close_connection
#
# def capture_frames(frame_queue):
#     cap = cv2.VideoCapture(0)
#     cap.set(3, 640)  # Set width
#     cap.set(4, 480)  # Set height
#
#     if not cap.isOpened():
#         print("ERROR: Could not open the camera")
#         return
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame from camera.")
#             break
#         frame_queue.put(frame)
#
#     cap.release()
#
# def process_frames(frame_queue):
#     # Load the encoding file if it exists
#     encodeListKnown, empNames = [], []
#     if os.path.exists('EncodeFile.p'):
#         print("Loading The Encoding File.....")
#         with open('EncodeFile.p', 'rb') as file:
#             encodeListKnownWithNames = pickle.load(file)
#             encodeListKnown, empNames = encodeListKnownWithNames
#         print("Encoding File Loaded....")
#     else:
#         print("EncodeFile.p not found. Starting with an empty encoding list.")
#
#     connection = connect_to_database()
#     if connection:
#         create_table(connection)
#
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#
#             # Resize and process frame
#             frameS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
#             frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
#
#             # Detect faces and encodings
#             faceCurrFrame = face_recognition.face_locations(frameS)
#             encodeCurrFrame = face_recognition.face_encodings(frameS, faceCurrFrame)
#
#             face_detected = False
#             for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
#                 matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#                 faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#
#                 if matches and any(matches):
#                     if faceDis.size > 0:
#                         matchIndex = np.argmin(faceDis)
#                         if matches[matchIndex]:
#                             face_detected = True
#                             print("Known Face Detected.")
#                             capture_date = datetime.now().strftime("%d-%m-%y")
#                             capture_time = datetime.now().strftime("%I:%M:%S%p")
#                             print(f"Detected: {empNames[matchIndex]} at {capture_time} on {capture_date}")
#
#                             # Save the real-time image when the face is detected
#                             realtime_image_path = f'Employee_Images/{empNames[matchIndex]}.jpg'
#                             cv2.imwrite(realtime_image_path, frame)
#
#                             # Insert the detection event into the database
#                             insert_image_data(connection, empNames[matchIndex], realtime_image_path, face_detected)
#                             break
#                 else:
#                     print("No matches found.")
#                     print("No known face detected.")
#
#                     # Prompt for the user's name
#                     employee_name = input("New person detected. Please enter your name in FirstName_LastName format: ")
#
#                     # Save the new image
#                     image_counter = 1
#                     existing_files = os.listdir('Employee_Images')
#                     pattern = re.compile(rf'{employee_name}_(\d+)\.jpg')
#
#                     for filename in existing_files:
#                         match = pattern.match(filename)
#                         if match:
#                             number = int(match.group(1))
#                             if number >= image_counter:
#                                 image_counter = number + 1
#
#                     image_path = f'Employee_Images/{employee_name}_{image_counter}.jpg'
#                     cv2.imwrite(image_path, frame)
#                     print(f"Image saved as {image_path}")
#
#                     try:
#                         # Generate encodings and save the encoding file
#                         EG(image_path)
#                         if os.path.exists('EncodeFile.p'):
#                             print("Reloading The Encoding File.....")
#                             with open('EncodeFile.p', 'rb') as file:
#                                 encodeListKnownWithNames = pickle.load(file)
#                                 encodeListKnown, empNames = encodeListKnownWithNames
#                             print("Encoding File Reloaded....")
#                         else:
#                             print("Failed to generate EncodeFile.p")
#
#                         # Insert the captured image into the database
#                         insert_image_data(connection, employee_name, image_path, False)
#                     except Exception as e:
#                         print(f"An error occurred: {e}")
#                     break
#
#             if not face_detected:
#                 cv2.imshow('Webcam', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#
#     if connection:
#         close_connection(connection)
#     cv2.destroyAllWindows()
#
# # Create a queue for frames
# frame_queue = queue.Queue(maxsize=10)
#
# # Create threads for capturing and processing frames
# capture_thread = threading.Thread(target=capture_frames, args=(frame_queue,))
# process_thread = threading.Thread(target=process_frames, args=(frame_queue,))
#
# # Start the threads
# capture_thread.start()
# process_thread.start()
#
# # Wait for the threads to finish
# capture_thread.join()
# process_thread.join()

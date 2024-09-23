from flask import Flask,render_template,Response,request,redirect,url_for
import cv2
import os
import re
import pickle
import face_recognition
import numpy as np
from EncodeGenrator import *
from datetime import datetime
from db_connection import connect_to_database, create_table, insert_image_data, close_connection
import threading
import queue
import time

#Initialize the Flask app
app = Flask(__name__)

#Global variable to store the VideoCapture object
cap = None

#Create a queue for user input
user_input_queue = queue.Queue()

def user_input_thread(user_input_queue):
    """Thread function to handle user input."""
    # Prompt for employee name and put it in the queue
    #print(f"Queue size before adding: {user_input_queue.qsize()}")
    employee_name = input("Please enter your name in FirstName_LastName format: ")
    user_input_queue.put(employee_name)
    #print(f"Queue size after adding: {user_input_queue.qsize()}")
    #print(f"DEBUG: Employee name '{employee_name}' added to queue.")
    time.sleep(0.1)

@app.route('/')
def index():
     #Modes: Active,New User,Marked,Already Marked
     mode = "active"
     return render_template('main2.html',mode=mode)

# @app.route('/process_image', methods=['POST'])
# def process_image():
#     # Your existing logic to get `employee_name`, `realtime_image_path`, and `face_detected`
#     mode = insert_image_data(connection, employee_name, realtime_image_path, face_detected)
#     return redirect(url_for('main2.html', mode=mode))


@app.route('/video_feed')
def video_feed():
    return Response(capture_image2(), mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_image2():
    global cap
    cap = cv2.VideoCapture(0)
    #cap.set(3, 640)  # Set width
    #cap.set(4, 480)  # Set height

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

    # Create Employee Images directory if it doesn't exist
    if not os.path.exists('Employee_Images'):
        os.makedirs('Employee_Images')

    image_counter = 1
    existing_files = os.listdir('Employee_Images')
    pattern = re.compile(rf'(.+?)_(\d+)\.jpg')

    for filename in existing_files:
        match = pattern.match(filename)
        if match:
            number = int(match.group(2))
            if number >= image_counter:
                image_counter = number + 1

    # Connect to the database and create table if it doesn't exist
    connection = connect_to_database()
    if connection:
        create_table(connection)

    message_printed = False
    employee_name = None
    input_thread_started = False

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        # Resizing the image to reduce computational power
        frameS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

        # Detect faces and encodings in the current frame
        faceCurrFrame = face_recognition.face_locations(frameS)
        encodeCurrFrame = face_recognition.face_encodings(frameS, faceCurrFrame)

        face_detected = False
        for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            if faceDis.size > 0:
                matchIndex = np.argmin(faceDis)

                # Define a custom threshold for strict matching
                threshold = 0.5

                if matches[matchIndex] and faceDis[matchIndex] <= threshold:
                    face_detected = True
                    print("Known Face Detected.")
                    capture_date = datetime.now().strftime("%d-%m-%y")
                    capture_time = datetime.now().strftime("%I:%M:%S%p")
                    print(f"Detected: {empNames[matchIndex]} at {capture_time} on {capture_date}")

                    realtime_image_path = f'Employee_Images/{empNames[matchIndex]}_realtime.jpg'
                    cv2.imwrite(realtime_image_path, frame)

                    insert_image_data(connection, empNames[matchIndex], realtime_image_path, face_detected)
                    os.remove(realtime_image_path)

                else:
                    face_detected = False

                if face_detected:
                    top,right,bottom,left = faceLoc
                    top,right,bottom,left = top*4,right*4,bottom*4,left*4
                    cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
                    cv2.putText(frame,empNames[matchIndex],(left+6,top-20),cv2.FONT_HERSHEY_TRIPLEX,0.75,(255,0,0),1)

        # # For Debugging
        # cv2.imshow('FaceAttendance', frame)
        # cv2.waitKey(1) & 0xFF

        # Encode the frame to PNG format
        ret, buffer = cv2.imencode('.png', frame)
        frame_bytes = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    if request.method == 'POST':
        employee_name = request.form.get('employee_name')
        global cap
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not cap.isOpened():
                print("ERROR: Could not open the camera")
                return
            if not os.path.exists('Employee_Images'):
                os.makedirs('Employee_Images')

            image_counter = 1
            existing_files = os.listdir('Employee_Images')
            pattern = re.compile(rf'(.+?)_(\d+)\.jpg')
            try:
                realtime_image_path = f'Employee_Images/{employee_name}_{image_counter}.jpg'
                cv2.imwrite(realtime_image_path, frame)
                print(f"Image saved as {realtime_image_path}")
                break
            except:
                realtime_image_path = ""
                pass
            image_counter += 1

        print(realtime_image_path)
        # Generate encodings and save the encoding file
        try:
            EG(realtime_image_path)
        except Exception as e:
            print(e)
            pass
        # Reload the encoding file
        if os.path.exists('EncodeFile.p'):
            print("Reloading The Encoding File.....")
            with open('EncodeFile.p', 'rb') as file:
                encodeListKnownWithNames = pickle.load(file)
                encodeListKnown, empNames = encodeListKnownWithNames
            print("Encoding File Reloaded....")
        else:
            print("Failed to generate EncodeFile.p")
    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print('encoding generated....!!!!')
    # return Response(capture_image2(), mimetype='multipart/x-mixed-replace; boundary=frame')
    mode = 'active'
    return render_template('main2.html', mode=mode)

if __name__ == "__main__":

    #capture_image()

    #Start the Flask application
    app.run(debug=True)



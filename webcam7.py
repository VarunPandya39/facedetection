from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import re
import pickle
import face_recognition
import numpy as np
from EncodeGenrator import EG
from datetime import datetime
from db_connection import connect_to_database, create_table, insert_image_data, close_connection

# Initialize the Flask app
app = Flask(__name__)

# Global variables for VideoCapture and database connection
cap = None
connection = None

@app.route('/')
def index():
    # Modes: Active, New User, Marked, Already Marked
    mode = "active"
    return render_template('main2.html', mode=mode)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # Encode the frame to PNG format
        ret, buffer = cv2.imencode('.png', frame)
        frame_bytes = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')

# Load the encoding file if it exists
encodeListKnown, empNames = [], []
if os.path.exists('EncodeFile.p'):
    with open('EncodeFile.p', 'rb') as file:
        encodeListKnownWithNames = pickle.load(file)
        encodeListKnown, empNames = encodeListKnownWithNames

if connection is None:
    connection = connect_to_database()
    if connection:
        create_table(connection)

def process_frame(cap,encodeListKnown,empNames,connection):

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        # Detect faces and encodings in the current frame
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

                if face_detected:
                    top, right, bottom, left = faceLoc
                    top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, empNames[matchIndex], (left + 6, top - 20), cv2.FONT_HERSHEY_TRIPLEX,
                                0.75, (255, 0, 0), 1)

        # Encode the frame to PNG format
        ret, buffer = cv2.imencode('.png', frame)
        frame_bytes = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')


    return

@app.route('/capture_image', methods=['POST'])
def capture_image():
    global cap
    global connection

    if cap is None:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return "ERROR: Could not open the camera", 500

    # Load the encoding file if it exists
    encodeListKnown, empNames = [], []
    if os.path.exists('EncodeFile.p'):
        with open('EncodeFile.p', 'rb') as file:
            encodeListKnownWithNames = pickle.load(file)
            encodeListKnown, empNames = encodeListKnownWithNames

    # Capture an image
    ret, frame = cap.read()

    if not ret:
        return "Error: Failed to capture frame from camera.", 500

    employee_name = request.form.get('employee_name')
    if employee_name:
        image_counter = 1
        existing_files = os.listdir('Employee_Images')
        pattern = re.compile(rf'(.+?)_(\d+)\.jpg')

        for filename in existing_files:
            match = pattern.match(filename)
            if match:
                number = int(match.group(2))
                if number >= image_counter:
                    image_counter = number + 1

        realtime_image_path = f'Employee_Images/{employee_name}_{image_counter}.jpg'
        cv2.imwrite(realtime_image_path, frame)
        print(f"Image saved as {realtime_image_path}")
        image_counter += 1

        # Generate encodings and save the encoding file
        EG(realtime_image_path)

        # Reload the encoding file
        if os.path.exists('EncodeFile.p'):
            with open('EncodeFile.p', 'rb') as file:
                encodeListKnownWithNames = pickle.load(file)
                encodeListKnown, empNames = encodeListKnownWithNames

        #Process the captured frame to detect the new employee
        process_frame(cap,encodeListKnown,empNames,connection)

    if cap:
        cap.release()
        cv2.destroyAllWindows()

    if connection:
        close_connection(connection)

    return redirect(url_for('main2.html'))

#    return "No employee name provided.", 400

# @app.teardown_appcontext
# def teardown_appcontext(exception):
#     global cap
#     global connection


if __name__ == "__main__":
    app.run(debug=True)


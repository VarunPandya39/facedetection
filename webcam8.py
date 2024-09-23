from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import os
#import re
import pickle
import face_recognition
import numpy as np
from EncodeGenrator import EG
from datetime import datetime
from db_connection import get_encoding_from_db, connect_to_database, create_table, insert_image_data,load_encodings_from_db,close_connection, check_attendance, insert_attendance

# Initialize the Flask app
app = Flask(__name__)

# Global variables for VideoCapture and database connection
cap = cv2.VideoCapture(0)
connection = None

@app.route('/')
def index():
    mode = 'new_user'
    return render_template('main2.html',mode=mode)

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.get_json()
    face_encoding = np.array(data.get('face_encoding'))  # Assuming face encoding is passed in JSON
    realtime_image_path = data.get('realtime_image_path')

    # Fetch employee details based on face encoding
    emp_id, employee_name = get_encoding_from_db(connection, face_encoding)  # Implement this function in db_connection.py

    if emp_id is None:
        # If no matching face encoding found, return status for a new employee
        return jsonify(status='new_user', message='No matching face encoding found'), 404

    # Check if attendance already marked for the day for the emp_id
    attendance_status = check_attendance(connection, emp_id)  # Implement this function in db_connection.py

    if attendance_status == 'present':
        # If attendance already marked for today, return appropriate message
        return jsonify(status='already_marked', message=f'Attendance already marked for {employee_name}'), 200
    elif attendance_status == 'absent':
        # If not marked, insert the attendance
        insert_attendance(emp_id, realtime_image_path)  # Implement this function in db_connection.py
        return jsonify(status='marked', message=f'Attendance marked for {employee_name}'), 200
    else:
        return jsonify(status='no_image', message='No valid image found'), 400

def process_frame():
    connection = connect_to_database()
    empNames, existing_encodings = load_encodings_from_db()  # Load encodings here

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
            matches = face_recognition.compare_faces(existing_encodings, encodeFace)
            faceDis = face_recognition.face_distance(existing_encodings, encodeFace)

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

                    # Fetch emp_id based on faceEncoding
                    cursor = connection.cursor()
                    faceEncoding_blob = existing_encodings[matchIndex].tobytes()  # Convert encoding to blob format

                    print(faceEncoding_blob)
                    cursor.execute("SELECT id FROM employee WHERE faceEncoding = %s", (faceEncoding_blob,))
                    emp_id_result = cursor.fetchone()
                    cursor.close()

                    if emp_id_result:
                        emp_id = emp_id_result[0]
                        realtime_image_path = f'Employee_Images/{empNames[matchIndex]}_realtime.jpg'
                        cv2.imwrite(realtime_image_path, frame)
                        dict_value = insert_image_data(connection, emp_id, realtime_image_path, face_detected)
                        print(f'status value==  {dict_value}')
                        os.remove(realtime_image_path)
                    else:
                        print("Employee ID not found for face encoding.")

                # Draw rectangle and label for detected face
                top, right, bottom, left = faceLoc
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                empName = empNames[matchIndex].split('_')[0]
                cv2.putText(frame, empName, (left + 6, top - 20), cv2.FONT_HERSHEY_TRIPLEX,
                            0.75, (255, 0, 0), 1)

        # Handle case for new user (if no faces match)
        if not face_detected:
            print("New user detected. Prompting for input.")
            # Add logic to handle new user form/input here

        # Encode the frame to PNG format
        ret, buffer = cv2.imencode('.png', frame)
        frame_bytes = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')

        # For Debug
        # cv2.imshow('FaceAttendance', frame)

@app.route('/capture_image', methods=['POST'])
def capture_image():
    global cap
    global connection
    global empNames, existing_encodings

    try:
        # Load encodings from the database
        empNames, existing_encodings = load_encodings_from_db()
        print(f"Loaded {len(existing_encodings)} encodings from the database.")
        print(f"Employee Names: {empNames}")

        # Capture the frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame from camera.")
            return "Error: Failed to capture frame from camera.", 500

        employee_name = request.form.get('employee_name')
        if not employee_name:
            print("Error: No employee name received.")
            return "Error: Employee name not provided.", 400

        # Save the captured image
        image_path = f'Employee_Images/{employee_name}.jpg'
        cv2.imwrite(image_path, frame)
        print(f"Image saved as {image_path}")

        # Generate encodings
        try:
            encodings = EG(image_path)  # Assuming EG is your encoding generation function
            print(f"Encodings generated: {encodings}")
        except Exception as e:
            print(f"Error generating encodings: {e}")
            return "Error: Failed to generate encodings.", 500

        # Reload the encodings
        empNames, existing_encodings = load_encodings_from_db()

        # Perform face detection
        frameS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

        faceCurrFrame = face_recognition.face_locations(frameS)
        encodeCurrFrame = face_recognition.face_encodings(frameS, faceCurrFrame)

        face_detected = False
        for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
            matches = face_recognition.compare_faces(existing_encodings, encodeFace)
            faceDis = face_recognition.face_distance(existing_encodings, encodeFace)
            print(f"Comparing face encoding with existing encodings, face distances: {faceDis}")

            if matches and len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)
                threshold = 0.5
                if matches[matchIndex] and faceDis[matchIndex] <= threshold:
                    face_detected = True
                    print(f"Matched with {empNames[matchIndex]} (distance: {faceDis[matchIndex]})")

                    # Save the image and insert the data
                    realtime_image_path = f'Employee_Images/{empNames[matchIndex]}_realtime.jpg'
                    cv2.imwrite(realtime_image_path, frame)
                    try:
                        insert_image_data(connection, empNames[matchIndex], realtime_image_path, face_detected)
                        print(f"Image data inserted for {empNames[matchIndex]}")
                    except Exception as e:
                        print(f"Error inserting image data for {empNames[matchIndex]}: {e}")
                    os.remove(realtime_image_path)

    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An error occurred during image capture.", 500

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)

    if cap.isOpened():
        cap.release()

    if connection:
        close_connection(connection)

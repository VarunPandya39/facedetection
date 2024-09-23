# import cv2
# import pickle
# import face_recognition
# import numpy as np
# import datetime
#
# def load_encodings(file_path='EncodeFile.p'):
#     """Load face encodings and names from a file."""
#     print("Loading The Encoding File.....")
#     with open(file_path, 'rb') as file:
#         encodeListKnownWithNames = pickle.load(file)
#         #encodeListKnownWithNames = load_encodings()
#     print("Encoding File Loaded....")
#     return encodeListKnownWithNames
#
# def detect_faces(frame,encodeListKnown,empNames):
#
#         #Resizing the image so it doesnt take a lot of computational power
#         frameS = cv2.resize(frame,(0,0),None,0.25,0.25)
#         frameS = cv2.cvtColor(frameS,cv2.COLOR_BGR2RGB)
#
#         #Feeding the value to the face recognition system so it detects and gives us an output
#         #so we feed in the faces and encodings in the current frame
#         #we compare the faceEncodings of the new faces in the current frame and compare it with the previous encodings
#         faceCurrFrame = face_recognition.face_locations(frameS)
#         encodeCurrFrame = face_recognition.face_encodings(frameS,faceCurrFrame)
#
#         # # Initialize a flag to indicate whether a known face is detected
#         # # known_face_detected = False
#         # detected_faces = []
#         # detected_names = []
#
#         print("Detected faces:", len(faceCurrFrame))  # Debug: Number of faces detected
#
#         #Looping through the currFrame encodings and then comparing it with the genrated encodings
#         for encodeFace,faceLoc in zip(encodeCurrFrame,faceCurrFrame):
#             matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#             faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
#             #print("Matches", matches)
#             #print("FaceDis", faceDis)
#
#             matchIndex = np.argmin(faceDis)
#
#             #print(matchIndex)
#
#             if matches[matchIndex]:
#                detect_faces.append(matchIndex)
#             # if faceDis.size > 0:  # Check if faceDis is not empty
#             #     matchIndex = np.argmin(faceDis)
#             #     if matches and len(matches) > matchIndex and matches[matchIndex]:  # Additional checks
#             #         # detected_faces.append(matchIndex)
#             #         # detected_names.append(detected_names[matchIndex])
#             #         detected_faces.append(matchIndex)
#             #         detected_name = empNames[matchIndex]  # Use empNames here
#             #         detected_names.append(detected_name)
#
#                     capture_time = datetime.now().strftime("%Y-%m-%d,%H:%M:%S%p")
#                     # Print the detected name and capture time
#                     print(f"Detected {detected_name} at {capture_time}")
#
#         # print("Detected faces names:", detected_names)  # Debug: Names of detected faces

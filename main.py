import os
import face_recognition as fr
import cv2
import numpy as np
import csv
from datetime import datetime
import pickle


if not os.path.exists('students_encoding_cache/cache.pkl'):
    print('Load the students data using load_students_data.py')
    exit()
cache = pickle.load(open('students_encoding_cache/cache.pkl','rb'))
known_encodings = cache['encodings']
known_names = cache['names']

cam_url = "http://192.168.137.129:4747/video"
capture_video = cv2.VideoCapture(0)
attendance = {}


while True:
    ret, frame = capture_video.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = fr.compare_faces(known_encodings, face_encoding,tolerance=0.454)
        face_distances = fr.face_distance(known_encodings, face_encoding)

        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]
        else:
            name = 'unknown'

        if name!='unknown' and name not in attendance:
            now = datetime.now().strftime("%H:%M:%S")
            attendance[name]=now
            print(f"{name} marked present at {now}")

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.namedWindow('attendance-system', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('attendance-system', frame)

    if cv2.waitKey(1) & 0xFF == 27 :
        break

capture_video.release()
cv2.destroyAllWindows()

with open("attendance.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Time"])
    for name, time in attendance.items():
        writer.writerow([name, time])

print("Attendance saved to attendance.csv")
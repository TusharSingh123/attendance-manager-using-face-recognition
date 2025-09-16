import os
import pickle
import face_recognition as fr


STUDENT_DIR = "students"
cache_location = f"students_encoding_cache/cache.pkl"

known_encodings = []
known_names = []

for student in sorted(os.listdir(STUDENT_DIR)):
    student_folder = os.path.join(STUDENT_DIR,student)
    if not os.path.isdir(student_folder):
        continue

    for image in os.listdir(student_folder):
        image_path = os.path.join(student_folder,image)
        img = fr.load_image_file(image_path)
        encodings =  fr.face_encodings(img)
        if len(encodings)>0:
            known_encodings.append(encodings[0])
            known_names.append(student)


cache_obj = {"encodings": known_encodings,"names": known_names}
pickle.dump(cache_obj,open(cache_location,"wb"))
print(f"Loading encodings for {len(set(known_names))} students ")
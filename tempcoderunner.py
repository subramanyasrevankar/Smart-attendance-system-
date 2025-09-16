import cv2 
import face_recognition 
import pickle
import os 
#File path to store face encodings and names 
face_data_file='face_data.pkl'
#load existing data or initialize 
face_data=pickle.load(open(face_data_file,'rb')) if os.path.exists(face_data_file) else {'encodings': [], 'names': []}

def capture_face_and_name():
    """Capture a face and save it with a name"""
cap=cv2.VideoCapture(0)
print("Press 's' to save your face')
while True:
 ret,img=cap.read() 
 if not ret:
   print('Failed to capture')
   break
   cv2.imshow('Capture Face',img)
   if cv2.waitKey(1) & 0xFF==ord('s):
        img_small = cv2.cvtColor(cv2.resize(img, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(img_small)   #a list of encoding vectors (one per detected face). If no faces found → empty list
                    if encodings:
                         name=input("Enter your name")
                         face_data['encodings'].append(encodings[0])   #Takes the first face encoding found (encodings[0]) and appends it to the 'encodings' list in face_data.
                         face_data['names'].append(name)
                        with open(face_data_file, 'wb') as f:
                         pickle.dump(face_data, f)
                        print(f"Saved face for {name}")
                    else:
                       print("No face detected. Try again.")

                    elif cv2.waitKey(1) & 0xFF == ord('q'):
                  break
                  cap.release()
                  cv2.destroyAllWindows()

def recognize_face():
      cap = cv2.VideoCapture(0)
      if not cap.isOpened():
        print("Cannot open webcam")
        return
        while True:
         ret, img = cap.read()
        if not ret:
            print("Failed to capture image")
            break
             img_small = cv2.cvtColor(cv2.resize(img, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
                     face_locs = face_recognition.face_locations(img_small)
                     encodings = face_recognition.face_encodings(img_small, face_locs)
                             for encoding, loc in zip(encodings, face_locs):
                                            matches = face_recognition.compare_faces(face_data['encodings'], encoding)
                                            distances = face_recognition.face_distance(face_data['encodings'], encoding)
                                                      if True in matches:
                                                            best_match = distances.argmin()
                                                            name = face_data['names'][best_match]
                                                            confidence = round((1 - distances[best_match]) * 100, 2)
                                                             y1, x2, y2, x1 = [coord * 4 for coord in loc]
                                                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                                            cv2.putText(img, f"{name} ({confidence}%)", (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                                                    else:
                                                         print("Stranger detected")
                                                            cv2.imshow("Recognize Face", img)
                                                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                                                             break
                                                            cap.release()
                                                            cv2.destroyAllWindows()
                                                            capture_face_and_name()
                                                             recognize_face()




                                          





                                                            













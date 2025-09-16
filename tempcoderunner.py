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
                    encodings = face_recognition.face_encodings(img_small)
                    


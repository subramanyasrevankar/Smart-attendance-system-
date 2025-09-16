import cv2
import numpy as np 
import os 
import pickle 
from datetime import datetime 
import openpyxl 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder 
#File Paths 
face_data_file='hog_svm_face_data.pkl'
attendance_file='attendance.xlsx'
model_file='hog_svm_face_model.pkl' 

#Initialize or create the Excel file for attendance 
if not os.path.exists(attendance_file):
  wb=openpyxl.Workbook()
  ws=wb.active #gets the work sheet
  ws.title='Attendance'
  ws.append(['Name','Date','Time','Status']) 
  wb.save(attendance_file)

def mark_attendance(name):
  wb.openpyxl.load_workbook(attendance_file)
  ws=wb.active
  now=datetime.now()
  date_str=now.strftime("%Y-%m-%d")
  time_str=now.strftime("%H:%M:%S")
  ws.append([name,date_str,time_str,"Present"])
  wb.save(attendance_file)

#Extract the  HOG features (Histogram Oriented Gradients)->think it is like a shape detector
def extract_hog_features(image):
  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  hog=cv2.HOGDescriptor()
  hog_features=hog.compute(gray) # extract the hog features and compute it in the 1d 
  return hog_features.flatten() #helps to flatten 

def train_svm_model():
  if not os.path.exists(face_data_file):
      print("No face data found.Register Faces first")
      return 
    with open(face_data_file,'rb) as f:
              data =pickle(f) #helps to load the binary file
       features=data['features']
       names=data['names']
       #Ensure all the features have the same shape 
        max_length=max(len(feature) for feature in features)  #calculates all length 
        padded_features=[np.pad(feature,(0,max_length-len(feature)),'constant')for feature in features]
        features_array=np.array(padded_features)
        names_array=np.array(names) #making ir i 2 d array makes it easy (ml)
        #Encode the labels 
        label_encoder=LabelEncoder()
        labels=labels_encoder.fit_transform(names_array)
       #Train the SVM model
        svm_model=SVC(kernel='linear',probability=True)  # it allows to calculates the probability  support vector machine which helps to classify the tasks
        svm_model.fit(features_array,labels)
        with open(model_file,'wb') as f:
          pickle.dump((svm_model,label_encoder),f) #Saves both the trained SVM model and the label encoder in a file called hog_svm_face_model.pkl.(later can be called with the name pickle.dump)
        print("SVM model trained and saved")

#capture the face with name
def capture_face_and_name():
  cap=cv2.VideoCapture(0)
  face_data={'features':[],'names':[]}
  if os.path.exists(face_data_file):
    with open(face_data_file,'rb') as f:
      face_data=pickle.load(f)
      print("Press 's' to save your face")
   while True:
     _,img=cap.read()
     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
        face_img=img[y:y+h,x:x+w]
        hog_features=extract_hog_features(face_img)
        
        # Ensure all features have the same length
        if face_data['features']:
            max_length = max(len(feature) for feature in face_data['features'])
            hog_features = np.pad(hog_features, (0, max(0, max_length - len(hog_features))), 'constant')
					name=input('Enter your name')
					face_data['features'].append(hog_features)
					face_data['names'].apppend(name)
					with open(face_data_file.'wb') as f:
              pickle.dump(face_data,f)
					print(f"Face data saved for {name}.")
					cv2.imshow("Capture Face",img) #moved before break
					break 
		if cv2.waitKey(1) & 0xFF==ord('s):
					break
		cap.release()
	  cap.destroyAllWindows()
   
   
def recognize_face():
    if not os.path.exists(model_file):
	      print("Model not found.Train it first")
	     return 
	with open('model_file','rb')as f"
     svm_model,label_encoder=pickle_load(f)
	 cap=cv2.VideoCapture(0)
      if not cap.isOpened():
	     print("Cannot open Webcam")
	     return 
	  face_cascade=cv2.CascadeClassifier(cv2.data,haarcascades + 'haarcascade_frontalface_default.xml')
       seen=set()
	   while True:
	      ret,frame=cap.read()
	      if not ret:
	           break
		   gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 1.3-scale factor (how much image size is reduced at each scale)->5 ->min 5 detections to consider it as a valid face
			  for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            feat = extract_hog_features(face_img)  # the 1d vector 
            expected_len = svm_model.support_vectors_.shape[1]  # it is the trained support classifier 
            if len(feat) != expected_len:
                feat = np.pad(feat, (0, max(0, expected_len - len(feat))), 'constant')
            probs = svm_model.predict_proba([feat])[0]  # gives probability for each class
            idx = np.argmax(probs)  # best guess
            conf = probs[idx]
            if conf > 0.6:
                name = label_encoder.inverse_transform([idx])[0]  # get the name using the inverse transform 
                cv2.putText(frame, f"{name} {conf:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if name not in seen:
                    mark_attendance(name)
                    seen.add(name)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

while True:
 print("\n Choose an option")
 print("1.Register a new Face")
 print("2.Train SVM model')
 print('3.Recognize and mark attendance")
 print('4.Exit ')

 if choice == '1':
        capture_face_and_name()
    elif choice == '2':
        train_svm_model()
    elif choice == '3':
        recognize_face()
    elif choice == '4':
        print('Exiting the program')
        break
    else:
        print('Invalid choice enter again:')


       
  
  
  

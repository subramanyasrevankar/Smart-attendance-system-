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


       
  
  
  

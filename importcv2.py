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
  date_str=now.strftime(
  
  
  

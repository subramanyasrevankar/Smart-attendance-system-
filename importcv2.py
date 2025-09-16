import cv2
import numpy as np
import os
import pickle
import openpyxl
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# -----------------------
# File Paths
# -----------------------
face_data_file = 'hog_svm_face_data.pkl'  # File to store captured faces (features + names)
attendance_file = 'attendance.xlsx'       # Excel file to store attendance
model_file = 'hog_svm_face_model.pkl'     # Trained SVM model + label encoder

# -----------------------
# Initialize Attendance File (if not exists)
# -----------------------
if not os.path.exists(attendance_file):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Attendance'
    ws.append(['Name', 'Date', 'Time', 'Status'])  # Columns for attendance
    wb.save(attendance_file)

# -----------------------
# Mark Attendance in Excel
# -----------------------
def mark_attendance(name):
    """Append name, date, time, and status to attendance.xlsx"""
    wb = openpyxl.load_workbook(attendance_file)
    ws = wb.active
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    ws.append([name, date_str, time_str, "Present"])
    wb.save(attendance_file)

# -----------------------
# Extract HOG Features
# -----------------------
def extract_hog_features(image):
    """
    Convert image to grayscale and extract Histogram of Oriented Gradients (HOG) features.
    HOG works like a shape + edge detector.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)  # Returns a column vector of features
    return hog_features.flatten()  # Flatten to 1D array for ML model

# -----------------------
# Train SVM Model
# -----------------------
def train_svm_model():
    """Train SVM model using stored face data and save the trained model."""
    if not os.path.exists(face_data_file):
        st.warning("⚠ No face data found. Please register faces first.")
        return

    # Load face data (features + names)
    with open(face_data_file, 'rb') as f:
        data = pickle.load(f)

    features = data['features']
    names = data['names']

    # Ensure all feature vectors have the same length (padding if needed)
    max_length = max(len(feature) for feature in features)
    padded_features = [np.pad(feature, (0, max_length - len(feature)), 'constant')
                       for feature in features]
    features_array = np.array(padded_features)
    names_array = np.array(names)

    # Encode labels (convert names to numeric IDs)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(names_array)

    # Train SVM classifier
    svm_model = SVC(kernel='linear', probability=True)  # Linear SVM with probability output
    svm_model.fit(features_array, labels)

    # Save model + label encoder
    with open(model_file, 'wb') as f:
        pickle.dump((svm_model, label_encoder), f)

    st.success("✅ SVM model trained and saved successfully!")

# -----------------------
# Capture Face + Name
# -----------------------
def capture_face_and_name():
    """Capture faces from webcam, ask for name, store HOG features in pickle file."""
    cap = cv2.VideoCapture(0)
    face_data = {'features': [], 'names': []}

    # If file already exists, load previous data
    if os.path.exists(face_data_file):
        with open(face_data_file, 'rb') as f:
            face_data = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    st.info("Press 's' to save your face and 'q' to quit.")

    while True:
        ret, img = cap.read()
        if not ret:
            st.error("❌ Failed to access webcam.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            hog_features = extract_hog_features(face_img)

            # Match length with existing data (padding)
            if face_data['features']:
                max_length = max(len(feature) for feature in face_data['features'])
                hog_features = np.pad(hog_features, (0, max(0, max_length - len(hog_features))), 'constant')

            cv2.imshow("Register Face", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                name = st.text_input("Enter your name:")
                if name:
                    face_data['features'].append(hog_features)
                    face_data['names'].append(name)
                    with open(face_data_file, 'wb') as f:
                        pickle.dump(face_data, f)
                    st.success(f"✅ Face data saved for {name}")
                    break
            elif key & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------
# Recognize Faces + Mark Attendance
# -----------------------
def recognize_face():
    """Recognize faces using trained SVM model and mark attendance."""
    if not os.path.exists(model_file):
        st.warning("⚠ Model not found. Please train it first.")
        return

    # Load trained model + label encoder
    with open(model_file, 'rb') as f:
        svm_model, label_encoder = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    seen = set()  # To avoid duplicate attendance marking
    stframe = st.empty()  # Streamlit placeholder for live video feed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            feat = extract_hog_features(face_img)

            # Ensure feature length matches model input
            expected_len = svm_model.support_vectors_.shape[1]
            if len(feat) != expected_len:
                feat = np.pad(feat, (0, max(0, expected_len - len(feat))), 'constant')

            probs = svm_model.predict_proba([feat])[0]  # Class probabilities
            idx = np.argmax(probs)  # Best match
            conf = probs[idx]

            if conf > 0.6:  # Confidence threshold
                name = label_encoder.inverse_transform([idx])[0]
                cv2.putText(frame, f"{name} {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if name not in seen:
                    mark_attendance(name)
                    seen.add(name)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------
# Streamlit UI
# -----------------------
st.title("📸 Smart Attendance System (HOG + SVM)")

option = st.radio("Choose an option:", ("Register Face", "Train Model", "Recognize & Mark Attendance"))

if option == "Register Face":
    if st.button("Start Registration"):
        capture_face_and_name()

elif option == "Train Model":
    if st.button("Train SVM Model"):
        train_svm_model()

elif option == "Recognize & Mark Attendance":
    if st.button("Start Recognition"):
        recognize_face()

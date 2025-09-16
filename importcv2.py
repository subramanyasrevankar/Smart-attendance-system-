import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import openpyxl
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# File Paths
face_data_file = 'hog_svm_face_data.pkl'
attendance_file = 'attendance.xlsx'
model_file = 'hog_svm_face_model.pkl'

# Initialize or create the Excel file for attendance
if not os.path.exists(attendance_file):
    wb = openpyxl.Workbook()
    ws = wb.active  # gets the work sheet
    ws.title = 'Attendance'
    ws.append(['Name', 'Date', 'Time', 'Status'])
    wb.save(attendance_file)


def mark_attendance(name):
    wb = openpyxl.load_workbook(attendance_file)
    ws = wb.active
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    ws.append([name, date_str, time_str, "Present"])
    wb.save(attendance_file)


# Extract the HOG features (Histogram of Oriented Gradients)
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)  # extract the HOG features
    return hog_features.flatten()  # flatten to 1D vector


def train_svm_model():
    if not os.path.exists(face_data_file):
        print("No face data found. Register Faces first.")
        return

    with open(face_data_file, 'rb') as f:
        data = pickle.load(f)  # load face data
        features = data['features']
        names = data['names']

    # Ensure all features have the same shape
    max_length = max(len(feature) for feature in features)
    padded_features = [np.pad(feature, (0, max_length - len(feature)), 'constant') for feature in features]
    features_array = np.array(padded_features)
    names_array = np.array(names)

    # Encode the labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(names_array)

    # Train the SVM model
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(features_array, labels)

    # Save model + encoder
    with open(model_file, 'wb') as f:
        pickle.dump((svm_model, label_encoder), f)

    print("✅ SVM model trained and saved.")


def capture_face_and_name():
    cap = cv2.VideoCapture(0)
    face_data = {'features': [], 'names': []}

    if os.path.exists(face_data_file):
        with open(face_data_file, 'rb') as f:
            face_data = pickle.load(f)

    print("Press 's' to save your face")

    while True:
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            hog_features = extract_hog_features(face_img)

            # Ensure all features have the same length
            if face_data['features']:
                max_length = max(len(feature) for feature in face_data['features'])
                hog_features = np.pad(hog_features, (0, max(0, max_length - len(hog_features))), 'constant')

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Capture Face", img)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                name = input('Enter your name: ')
                face_data['features'].append(hog_features)
                face_data['names'].append(name)
                with open(face_data_file, 'wb') as f:
                    pickle.dump(face_data, f)
                print(f"✅ Face data saved for {name}")
                cap.release()
                cv2.destroyAllWindows()
                return


def recognize_face():
    if not os.path.exists(model_file):
        print("Model not found. Train it first.")
        return

    with open(model_file, 'rb') as f:
        svm_model, label_encoder = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    seen = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            feat = extract_hog_features(face_img)
            expected_len = svm_model.support_vectors_.shape[1]

            if len(feat) != expected_len:
                feat = np.pad(feat, (0, max(0, expected_len - len(feat))), 'constant')

            probs = svm_model.predict_proba([feat])[0]
            idx = np.argmax(probs)
            conf = probs[idx]

            if conf > 0.6:
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

        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


while True:
    print("\nChoose an option")
    print("1. Register a new Face")
    print("2. Train SVM model")
    print("3. Recognize and mark attendance")
    print("4. Exit")

    choice = input("Enter choice: ")

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
        print('Invalid choice, enter again:')

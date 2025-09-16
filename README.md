🧠 Smart Attendance System (Face Recognition Based)

This project is a face recognition–based smart attendance system built using Python, OpenCV, HOG features, and SVM.
It captures faces, trains a machine learning model, and automatically marks attendance in an Excel sheet.

📌 Features

✅ Face Detection & Recognition – Detects faces from the webcam and recognizes registered persons.
✅ Automatic Attendance Marking – Marks name, date, time, and status in an Excel file.
✅ HOG + SVM Model – Uses Histogram of Oriented Gradients (HOG) for feature extraction and Support Vector Machine (SVM) for classification.
✅ Data Storage – Saves face encodings using pickle for reusability.
✅ Easy to Extend – You can add new students by capturing their face data and retraining the model.

📂 Project Structure
smart_attendance_system/
│
├── importcv2.py              # Main Python script (capture, train, and recognize)
├── face_data/                # Folder to store captured face images
├── models/
│   └── svm_model.pkl         # Trained SVM model (auto-created after training)
├── attendance.xlsx           # Attendance sheet (auto-created on first run)
└── README.md                 # This file

⚙️ Requirements

Install the required libraries before running:

pip install opencv-python numpy scikit-learn openpyxl

🚀 How to Use
1️⃣ Step 1: Capture Face Data

Run the script to capture your face images:

python importcv2.py --capture


The script will open your webcam.

Press spacebar or follow the on-screen instructions to capture images.

Images will be saved in face_data/ with your name as a label.

2️⃣ Step 2: Train the Model

After capturing data for at least two different persons, train the SVM model:

python importcv2.py --train


This will:

Extract HOG features from all images.

Train an SVM classifier.

Save the model as svm_model.pkl.

3️⃣ Step 3: Recognize & Mark Attendance

Run the recognition script:

python importcv2.py --recognize


The webcam will open.

Detected faces will be matched against the trained model.

If recognized, their name, date, time, and status will be marked in attendance.xlsx.

🛠 How It Works (Under the Hood)

Face Detection
Uses OpenCV’s cv2.CascadeClassifier to detect faces in frames.

Feature Extraction
Converts faces to grayscale, resizes them, and computes HOG features (edge/orientation information).

Model Training
Trains an SVM classifier with the extracted HOG features and corresponding labels.

Recognition + Attendance
During recognition, it predicts the person’s name using the trained SVM model and writes their details in an Excel sheet.

⚠️ Common Issues
Problem	Solution
ValueError: The number of classes has to be greater than one	You only have one person’s data. Capture face data for at least 2 people before training.
Webcam not opening	Check if your camera is free (not used by another app).
Model not found	Train the model first (--train) before running recognition.
📌 Future Improvements

Add deep learning (CNN-based) face recognition for better accuracy.

Build a GUI for easier usage.

Integrate with database instead of Excel for better scalability.

Deploy on Raspberry Pi for real-world use.

# Face Recognition Attendance System

A full-stack Machine Learning application that automates attendance tracking using facial recognition. Built with Flask and Keras, this system identifies faces from a local database in real-time.

## 🚀 Features
* **Real-time Recognition**: Uses a MobileNetV2-based model for efficient face identification.
* **Automated Logging**: Automatically records attendance into a SQLite database.
* **Web Interface**: User-friendly dashboard to view attendance records and manage user data.
* **Local Training**: Ability to add new faces and update embeddings locally.

## 🛠️ Tech Stack
* **Backend**: Python, Flask
* **Machine Learning**: Keras, TensorFlow, MobileNetV2, OpenCV, NumPy
* **Database**: SQLite
* **Frontend**: HTML5, CSS3, JavaScript

## 📂 Project Structure
* `app.py`: Main Flask application logic and API routes.
* `model/`: Contains the pre-trained MobileNetV2 weights.
* `data/`: Stores user face images and `embeddings.pkl`.
* `database/`: SQLite database files for attendance logs.
* `static/` & `templates/`: Web assets and UI components.

## ⚙️ Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/zohaib-hassan-dev/face-recognition-attendance-.git](https://github.com/zohaib-hassan-dev/face-recognition-attendance-.git)
   cd face-recognition-attendance-

from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime
import base64

app = Flask(__name__)

# Load model (MobileNetV2 as feature extractor)
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Path setup
FACE_DIR = 'data/faces'
EMB_PATH = 'data/embeddings.pkl'
DB_PATH = 'database/attendance.db'
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs('database', exist_ok=True)

# Load saved embeddings
if os.path.exists(EMB_PATH):
    with open(EMB_PATH, 'rb') as f:
        embeddings = pickle.load(f)
else:
    embeddings = {}

# Load face cascade for detection (built-in to OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_table_columns(conn, table_name):
    """Return list of column names for a table (or [] if table doesn't exist)."""
    cursor = conn.cursor()
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    if cursor.fetchone() is None:
        return []
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cursor.fetchall()]
    return cols

def init_db():
    """
    Initialize DB and perform non-destructive migrations:
      - create tables if missing
      - add missing columns to attendance table (user_name, timestamp, status) if they are absent
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure users table exists (basic)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')

    # If attendance table does not exist, create it with the desired schema
    cols = get_table_columns(conn, 'attendance')
    if not cols:
        cursor.execute('''
            CREATE TABLE attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                timestamp DATETIME,
                status TEXT DEFAULT 'Present'
            )
        ''')
        conn.commit()
        print("Created 'attendance' table with columns: id, user_name, timestamp, status")
    else:
        # attendance exists — ensure required columns are present; if missing, add them
        required = {
            'user_name': "TEXT",
            'timestamp': "DATETIME",
            'status': "TEXT DEFAULT 'Present'"
        }
        for col_name, col_def in required.items():
            if col_name not in cols:
                # ALTER TABLE ADD COLUMN is non-destructive in SQLite
                alter_sql = f"ALTER TABLE attendance ADD COLUMN {col_name} {col_def}"
                cursor.execute(alter_sql)
                print(f"Added column '{col_name}' to attendance table.")
        conn.commit()

    conn.close()

# Call init on startup (migrates DB if needed)
init_db()

def log_attendance(name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Insert using only columns we expect to exist (they should, because init_db migrates)
    cursor.execute('INSERT INTO attendance (user_name, timestamp) VALUES (?, ?)', (name, timestamp))
    conn.commit()  # Fixed: was 'ss on.commit()'
    conn.close()

def get_attendance_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # If timestamp column exists, order by it; otherwise return by id desc
    cols = get_table_columns(conn, 'attendance')
    if 'timestamp' in cols:
        cursor.execute('SELECT user_name, timestamp FROM attendance ORDER BY timestamp DESC')
    else:
        cursor.execute('SELECT user_name, id FROM attendance ORDER BY id DESC')
    logs = cursor.fetchall()
    conn.close()
    return logs

def get_single_embedding(img):
    """Extract single embedding from image."""
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    face_crop = img[y:y+h, x:x+w]
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = cv2.resize(face_crop, (224, 224))
    face_crop = np.expand_dims(face_crop, axis=0).astype(np.float32)
    face_crop = tf.keras.applications.mobilenet_v2.preprocess_input(face_crop)
    embedding = model.predict(face_crop)
    return embedding.flatten()

def average_embeddings(frames):
    """Average multiple embeddings for robustness."""
    embs = []
    for frame in frames:
        emb = get_single_embedding(frame)
        if emb is not None:
            embs.append(emb)
    if not embs:
        return None
    return np.mean(embs, axis=0)

def decode_base64_image(data_url):
    """
    Accepts a data URL like "data:image/png;base64,...." and returns an OpenCV BGR image.
    """
    if not data_url:
        return None
    if ',' in data_url:
        header, encoded = data_url.split(',', 1)
    else:
        encoded = data_url
    try:
        data = base64.b64decode(encoded)
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # returns BGR
        return img
    except Exception as e:
        print("Error decoding base64 image:", e)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/recognize')
def recognize_page():
    return render_template('recognize.html')

@app.route('/register_user', methods=['POST'])
def register_user():
    name = request.form.get('name') or (request.json.get('name') if request.json else None)
    if not name:
        return jsonify({"status": "error", "message": "Missing 'name' field"}), 400

    image_data = request.form.get('image') or (request.json.get('image') if request.json else None)
    frames = []
    if image_data:
        frame = decode_base64_image(image_data)
        if frame is not None:
            frames.append(frame)
    # Fallback: server camera for 3 frames
    if not frames:
        cap = cv2.VideoCapture(0)
        for _ in range(3):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        if not frames:
            return jsonify({"status": "error", "message": "Camera not working"}), 400

    embedding = average_embeddings(frames)
    if embedding is None:
        return jsonify({"status": "error", "message": "No face detected in any frame! Try facing the camera."}), 400

    embeddings[name] = embedding
    with open(EMB_PATH, 'wb') as f:
        pickle.dump(embeddings, f)

    # Add user to DB if missing
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # User already exists, skip
    conn.close()

    return jsonify({"status": "ok", "message": f"User {name} registered successfully!"})

@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    image_data = request.form.get('image') or (request.json.get('image') if request.json else None)
    frames = []
    if image_data:
        frame = decode_base64_image(image_data)
        if frame is not None:
            frames.append(frame)
    # Fallback: server camera for 3 frames
    if not frames:
        cap = cv2.VideoCapture(0)
        for _ in range(3):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        if not frames:
            return jsonify({"status": "error", "message": "Camera not working"}), 400

    emb = average_embeddings(frames)
    if emb is None:
        return jsonify({"status": "error", "message": "No face detected in any frame! Position clearly."}), 400

    best_match = None
    best_score = 0.6  # Stricter threshold for better accuracy
    for name, saved_emb in embeddings.items():
        try:
            sim = cosine_similarity([emb], [saved_emb])[0][0]
        except Exception as e:
            print(f"Skipping {name}: error comparing embeddings:", e)
            continue
        if sim > best_score:
            best_match = name
            best_score = sim

    if best_match:
        log_attendance(best_match)  # Log to DB
        return jsonify({
            "status": "ok",
            "message": f"Recognized: {best_match} (confidence: {best_score:.2f}) - Attendance marked!",
            "name": best_match,
            "confidence": float(best_score)
        })
    else:
        return jsonify({"status": "no_match", "message": "No match found."}), 200

@app.route('/users')
def users():
    user_list = list(embeddings.keys())
    return render_template('users.html', users=user_list)

@app.route('/delete_user/<name>')
def delete_user(name):
    if name in embeddings:
        del embeddings[name]
        with open(EMB_PATH, 'wb') as f:
            pickle.dump(embeddings, f)

        # Remove from DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE name = ?', (name,))
        conn.commit()
        conn.close()
    return redirect(url_for('users'))

@app.route('/attendance')
def attendance():
    logs = get_attendance_logs()
    return render_template('attendance.html', logs=logs)

# --- Optional debug helper you can call to see schema in console ---
def print_db_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    for name, sql in cursor.fetchall():
        print("TABLE:", name)
        print(sql)
        print("---")
    conn.close()

if __name__ == '__main__':
    # for debugging, print schema on startup
    try:
        print_db_schema()
    except Exception as e:
        print("Could not print DB schema:", e)
    app.run(debug=True)
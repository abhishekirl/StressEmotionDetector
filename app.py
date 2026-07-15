from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import numpy as np
import threading
import base64
import os
import io
import librosa
import soundfile as sf
from tensorflow import keras
import sqlite3
import json
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'stress_detection_secret_key_2024'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'users.db')

# ─── Database Setup ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    # Combined sessions table — face + speech + combined result
    c.execute('''CREATE TABLE IF NOT EXISTS combined_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,

        face_stress_score REAL,
        face_stress_level TEXT,
        face_stress_color TEXT,
        face_stress_emoji TEXT,
        face_emotion      TEXT,

        speech_stress_score REAL,
        speech_stress_level TEXT,
        speech_stress_color TEXT,
        speech_stress_emoji TEXT,
        speech_emotion      TEXT,

        combined_score REAL,
        combined_level TEXT,
        combined_color TEXT,
        combined_emoji TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# ─── Load Models ───────────────────────────────────────────────────────────────
import pickle
face_model   = keras.models.load_model(os.path.join(BASE_DIR, 'models', 'face_emotion_model.keras'))
speech_model = keras.models.load_model(os.path.join(BASE_DIR, 'models', 'speech_emotion_model.keras'))

# Load scaler and label encoder (required for correct speech predictions)
with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    speech_scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb') as f:
    speech_le = pickle.load(f)

# ─── Labels ────────────────────────────────────────────────────────────────────
FACE_LABELS   = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
SPEECH_LABELS = {0:'angry',1:'calm',2:'disgust',3:'fear',4:'happy',5:'neutral',6:'sad',7:'surprise'}
STRESS_EMOTIONS_FACE   = ['Angry','Fear','Disgust','Sad']
STRESS_EMOTIONS_SPEECH = ['angry','fear','disgust','sad']

# ─── Face Detection Setup ──────────────────────────────────────────────────────
face_cascade         = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_emotion_counts  = {label: 0 for label in FACE_LABELS.values()}
face_total_frames    = 0
current_face_emotion = 'Detecting...'
camera_lock          = threading.Lock()

# ─── Helper Functions ──────────────────────────────────────────────────────────
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def extract_audio_features(audio_data, sr=22050):
    mfcc        = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    stft        = np.abs(librosa.stft(audio_data))
    chroma      = librosa.feature.chroma_stft(S=stft, sr=sr)
    mel         = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    zcr         = librosa.feature.zero_crossing_rate(audio_data)
    rms         = librosa.feature.rms(y=audio_data)
    sc          = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    sb          = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
    sr2         = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
    scontrast   = librosa.feature.spectral_contrast(S=stft, sr=sr)
    features = np.concatenate([
        np.mean(mfcc,axis=1), np.mean(delta_mfcc,axis=1), np.mean(delta2_mfcc,axis=1),
        np.mean(chroma,axis=1), np.mean(mel,axis=1), np.mean(zcr,axis=1),
        np.mean(rms,axis=1), np.mean(sc,axis=1), np.mean(sb,axis=1),
        np.mean(sr2,axis=1), np.mean(scontrast,axis=1),
    ])
    return features.astype(np.float32)

def calculate_stress(emotion_percentages, stress_emotions):
    # Weighted sum of stress-inducing emotions (as % of total)
    stress_weights = {
        'Angry':0.9,  'angry':0.9,
        'Fear':0.85,  'fear':0.85,
        'Sad':0.7,    'sad':0.7,
        'Disgust':0.65,'disgust':0.65,
        'Surprise':0.2,'surprise':0.2,
        'Neutral':0.1, 'neutral':0.1,
    }
    total_pct  = sum(emotion_percentages.values()) or 1.0
    stress_sum = 0.0
    for emo in stress_emotions:
        pct    = emotion_percentages.get(emo, 0)
        weight = stress_weights.get(emo, 0.5)
        stress_sum += (pct / total_pct) * weight * 100
    score = max(0.0, min(100.0, round(stress_sum, 2)))
    if score < 30:   return score, 'Low Stress',    '#4CAF50', '😌'
    elif score < 60: return score, 'Medium Stress', '#FF9800', '😟'
    else:            return score, 'High Stress',   '#F44336', '😰'

def score_to_level(score):
    score = round(score, 2)
    if score < 30:   return score, 'Low Stress',    '#4CAF50', '😌'
    elif score < 60: return score, 'Medium Stress', '#FF9800', '😟'
    else:            return score, 'High Stress',   '#F44336', '😰'

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─── DB Query Helpers ──────────────────────────────────────────────────────────
def get_user_sessions(username, limit=100):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id,
        face_stress_score, face_stress_level, face_stress_color, face_stress_emoji, face_emotion,
        speech_stress_score, speech_stress_level, speech_stress_color, speech_stress_emoji, speech_emotion,
        combined_score, combined_level, combined_color, combined_emoji, created_at
        FROM combined_sessions WHERE username=? ORDER BY created_at DESC LIMIT ?''', (username, limit))
    rows = c.fetchall()
    conn.close()
    sessions = []
    for r in rows:
        sessions.append({
            'id':r[0],
            'face_stress_score':r[1],'face_stress_level':r[2],'face_stress_color':r[3],
            'face_stress_emoji':r[4],'face_emotion':r[5],
            'speech_stress_score':r[6],'speech_stress_level':r[7],'speech_stress_color':r[8],
            'speech_stress_emoji':r[9],'speech_emotion':r[10],
            'combined_score':r[11],'combined_level':r[12],'combined_color':r[13],
            'combined_emoji':r[14],'created_at':r[15]
        })
    return sessions

def get_dashboard_stats(username):
    sessions = get_user_sessions(username)
    total = len(sessions)
    low  = sum(1 for s in sessions if s['combined_level'] == 'Low Stress')
    med  = sum(1 for s in sessions if s['combined_level'] == 'Medium Stress')
    high = sum(1 for s in sessions if s['combined_level'] == 'High Stress')
    return {'total':total,'low_stress':low,'medium_stress':med,'high_stress':high}

# ─── Auth Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        if not username or not password:
            return render_template('login.html', error='Please enter both fields!')
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=?', (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session['user'] = username
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid username or password!')
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        confirm  = request.form.get('confirm_password','').strip()
        if not username or not password:
            return render_template('register.html', error='All fields are required!')
        if len(username) < 3:
            return render_template('register.html', error='Username must be at least 3 characters!')
        if len(password) < 4:
            return render_template('register.html', error='Password must be at least 4 characters!')
        if password != confirm:
            return render_template('register.html', error='Passwords do not match!')
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('INSERT INTO users (username,password) VALUES (?,?)',
                      (username, generate_password_hash(password)))
            conn.commit(); conn.close()
            return render_template('login.html', success='Account created! You can now login.')
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists!')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# ─── Page Routes ───────────────────────────────────────────────────────────────
@app.route('/face')
@login_required
def face_detection():
    return render_template('face_detection.html', username=session.get('user'))

@app.route('/audio')
@login_required
def audio_detection():
    return render_template('audio_detection.html', username=session.get('user'))

@app.route('/result')
@login_required
def result():
    return render_template('result.html', username=session.get('user'))

@app.route('/dashboard')
@login_required
def dashboard():
    username = session.get('user')
    sessions = get_user_sessions(username)
    stats    = get_dashboard_stats(username)
    return render_template('dashboard.html', username=username, sessions=sessions, stats=stats)



@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ─── Face Detection API ────────────────────────────────────────────────────────
@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    global face_emotion_counts, face_total_frames, current_face_emotion
    data      = request.json
    img_data  = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr     = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces     = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    detected_emotion = 'No Face'
    face_boxes = []
    for (x,y,w,h) in faces:
        face_roi   = frame[y:y+h, x:x+w]
        processed  = preprocess_face(face_roi)
        prediction = face_model.predict(processed, verbose=0)
        idx        = np.argmax(prediction[0])
        detected_emotion = FACE_LABELS[idx]
        confidence = float(prediction[0][idx]) * 100
        face_emotion_counts[detected_emotion] += 1
        face_total_frames += 1
        current_face_emotion = detected_emotion
        face_boxes.append({'x':int(x),'y':int(y),'w':int(w),'h':int(h),'emotion':detected_emotion,'confidence':round(confidence,1)})
    percentages = {}
    if face_total_frames > 0:
        for label, count in face_emotion_counts.items():
            percentages[label] = round((count/face_total_frames)*100, 2)
    stress_score, stress_level, stress_color, stress_emoji = calculate_stress(percentages, STRESS_EMOTIONS_FACE)
    return jsonify({'faces':face_boxes,'current_emotion':detected_emotion,'percentages':percentages,
                    'stress_score':stress_score,'stress_level':stress_level,'stress_color':stress_color,
                    'stress_emoji':stress_emoji,'total_frames':face_total_frames})

@app.route('/api/reset_face', methods=['POST'])
def reset_face():
    global face_emotion_counts, face_total_frames, current_face_emotion
    face_emotion_counts  = {label: 0 for label in FACE_LABELS.values()}
    face_total_frames    = 0
    current_face_emotion = 'Detecting...'
    return jsonify({'status':'reset'})

@app.route('/api/save_face_session', methods=['POST'])
@login_required
def save_face_session():
    """Stores face result in flask session for later combined save."""
    data = request.json
    session['face_data'] = {
        'face_stress_score': data.get('stress_score'),
        'face_stress_level': data.get('stress_level'),
        'face_stress_color': data.get('stress_color'),
        'face_stress_emoji': data.get('stress_emoji'),
        'face_emotion':      data.get('dominant_emotion'),
    }
    return jsonify({'status':'saved'})

# ─── Audio Detection API ───────────────────────────────────────────────────────
@app.route('/api/analyze_audio', methods=['POST'])
def analyze_audio():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({'error':'No audio file'}), 400
    tmp_path = None
    try:
        import tempfile
        audio_bytes = audio_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            tmp.write(audio_bytes); tmp_path = tmp.name
        try:
            audio_data, sr = librosa.load(tmp_path, sr=22050, mono=True)
        except Exception:
            try:
                buf = io.BytesIO(audio_bytes)
                audio_data, sr = sf.read(buf)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=22050)
                sr = 22050
            except Exception:
                audio_data = np.random.randn(22050).astype(np.float32) * 0.01
                sr = 22050
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
        if len(audio_data) == 0:
            return jsonify({'error':'Empty audio. Please record again.'}), 400
        # Extract features, scale, then predict
        raw_features = extract_audio_features(audio_data, sr)
        scaled_features = speech_scaler.transform(raw_features.reshape(1, -1))
        features = scaled_features.reshape(1, 1, -1)
        prediction = speech_model.predict(features, verbose=0)
        probs      = prediction[0]
        # Use label_encoder classes for correct label mapping
        emotion_percentages = {speech_le.classes_[i]: round(float(probs[i])*100, 2) for i in range(len(probs))}
        dominant_idx     = np.argmax(probs)
        dominant_emotion = speech_le.classes_[dominant_idx]
        stress_score, stress_level, stress_color, stress_emoji = calculate_stress(emotion_percentages, STRESS_EMOTIONS_SPEECH)
        # Store in flask session
        session['speech_data'] = {
            'speech_stress_score': stress_score,
            'speech_stress_level': stress_level,
            'speech_stress_color': stress_color,
            'speech_stress_emoji': stress_emoji,
            'speech_emotion':      dominant_emotion,
        }
        return jsonify({'emotion_percentages':emotion_percentages,'dominant_emotion':dominant_emotion,
                        'stress_score':stress_score,'stress_level':stress_level,
                        'stress_color':stress_color,'stress_emoji':stress_emoji})
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
        return jsonify({'error': str(e)}), 500

# ─── Save Combined Session API ─────────────────────────────────────────────────
@app.route('/api/save_combined_session', methods=['POST'])
@login_required
def save_combined_session():
    """Called from result.html — saves complete face+speech+combined row to DB."""
    data     = request.json
    username = session['user']

    face_score   = data.get('face_stress_score')
    speech_score = data.get('speech_stress_score')

    # Combined = Face 40% + Speech 60%
    if face_score is not None and speech_score is not None:
        combined_raw = (face_score * 0.5) + (speech_score * 0.5)
    elif face_score is not None:
        combined_raw = face_score
    elif speech_score is not None:
        combined_raw = speech_score
    else:
        combined_raw = 0

    c_score, c_level, c_color, c_emoji = score_to_level(combined_raw)

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO combined_sessions (
            username,
            face_stress_score, face_stress_level, face_stress_color, face_stress_emoji, face_emotion,
            speech_stress_score, speech_stress_level, speech_stress_color, speech_stress_emoji, speech_emotion,
            combined_score, combined_level, combined_color, combined_emoji
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
            username,
            face_score, data.get('face_stress_level'), data.get('face_stress_color'),
            data.get('face_stress_emoji'), data.get('face_emotion'),
            speech_score, data.get('speech_stress_level'), data.get('speech_stress_color'),
            data.get('speech_stress_emoji'), data.get('speech_emotion'),
            c_score, c_level, c_color, c_emoji
        ))
        conn.commit(); conn.close()
        return jsonify({'status':'saved','combined_score':c_score,'combined_level':c_level,
                        'combined_color':c_color,'combined_emoji':c_emoji})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

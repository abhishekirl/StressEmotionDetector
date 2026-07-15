# Stress and Emotion Detection Using Audio and Facial Expressions

## Project Title & Brief Description

This is a Flask web application that looks at facial expressions from a webcam and emotions from a short voice recording. Based on those predictions, it shows an emotion-based stress result.

I made this project to understand how face and speech emotion models can be used in a web application. The result is only an experimental indication. It is not meant to diagnose stress or any medical condition.

## Live Demo Links

Application: [https://stress-detection-pmka.onrender.com/](https://stress-detection-pmka.onrender.com/)

## Features

- User registration, login, and logout
- Face-emotion detection using the webcam
- Speech-emotion detection from a recorded audio sample
- Emotion percentages and dominant emotion for both inputs
- Low, Medium, or High stress indication
- Combined face and speech result
- Dashboard for saved check-ins
- Accuracy page with training curves and confusion matrices

## Tech Stack

- Python 3.12: used for the backend.
- Flask: used for pages, routes, sessions, and user authentication.
- TensorFlow / Keras: used to load and run the trained models.
- OpenCV: used to detect faces and prepare face images for prediction.
- Librosa and SoundFile: used to read audio and extract audio features.
- NumPy and scikit-learn: used for processing data, feature scaling, and speech labels.
- SQLite: used to store user accounts and saved sessions.
- HTML, CSS, and JavaScript: used to build the interface and access the camera and microphone.
- Gunicorn and Render: used for deployment.

## Requirements

- Python 3.12
- Webcam access for face detection
- Browser permission for the camera and microphone
- FFmpeg on the system path for reliable WebM audio processing

Install the required Python packages:

```bash
pip install -r requirements.txt
```

On macOS, install FFmpeg with Homebrew:

```bash
brew install ffmpeg
```

On Windows, install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to the system PATH.

## How To Run Locally

Clone the repository and move into the folder:

```bash
git clone https://github.com/abhishekirl/StressEmotionDetector.git
cd StressEmotionDetector
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and start the app:

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in the browser.

The trained models, speech scaler, and label encoder are already included in this repository.

## Using the Application

1. Register an account and log in.
2. Open **Face Detection** and allow camera access.
3. Start detection to view the predicted face emotion and stress result.
4. Open **Audio Detection** and allow microphone access.
5. Record a short sample to view the predicted speech emotion and stress result.
6. Open the combined-result page after completing one or both checks.
7. Save the result to view it later on the dashboard.

## How the Stress Result Is Calculated

The application predicts emotions from face frames and speech features. It gives more weight to angry, fear, sad, and disgust emotions. The final score is limited to a range from 0 to 100.

| Score | Result |
| --- | --- |
| Below 30 | Low Stress |
| 30 to below 60 | Medium Stress |
| 60 and above | High Stress |

If both face and speech results are available, the app averages their stress scores for the combined result.

## Dataset Sources

- Face-emotion model: [FER2013 Facial Expression Recognition Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- Speech-emotion model: [RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/records/1188976)

The training notebooks in `models/` show how the included models were trained.

## Project Structure

```text
StressEmotionDetector/
├── app.py                         # Flask application and API logic
├── requirements.txt               # Python packages
├── render.yaml                    # Render deployment configuration
├── runtime.txt                    # Runtime information
├── python-version                 # Python version information
├── users.db                       # SQLite database used by the app
├── label_encoder.pkl              # Speech-emotion label encoder
├── scaler.pkl                     # Speech-feature scaler
├── models/
│   ├── face_emotion_model.keras   # Trained face-emotion model
│   ├── speech_emotion_model.keras # Trained speech-emotion model
│   ├── face_emotion.ipynb         # Face-model training notebook
│   └── speech_emotion.ipynb       # Speech-model training notebook
├── static/                        # Accuracy graphs and other static files
└── templates/                     # HTML templates used by Flask
    ├── home.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── face_detection.html
    ├── audio_detection.html
    ├── result.html
    ├── accuracy.html
    └── about.html
```

## Pages and Routes

| Page | Route | Purpose |
| --- | --- | --- |
| Home | `/` | Welcome page |
| Login | `/login` | Log in to an account |
| Register | `/register` | Create an account |
| Face Detection | `/face` | Detect emotion from webcam frames |
| Audio Detection | `/audio` | Detect emotion from a recorded voice sample |
| Result | `/result` | Show the combined result |
| Dashboard | `/dashboard` | View saved check-ins |
| Accuracy | `/accuracy` | View training graphs and confusion matrices |
| About | `/about` | Project information |

## Current Limitations

- The result is not a medical or clinical assessment.
- Face predictions can be affected by lighting, camera angle, and image quality.
- Speech predictions can be affected by background noise, microphone quality, accents, and recording length.
- The models may not work equally well for every person or environment.
- SQLite is enough for this student project but is not intended for a large production application.

## Next Steps

- Improve face detection in low-light conditions and side angles
- Add a way for users to delete saved sessions
- Move the Flask secret key into an environment variable for deployment
- Add tests for the main routes and APIs
- Try the models on more data and real-world recordings

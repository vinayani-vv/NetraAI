<<<<<<< HEAD
# 🛡️ Netra AI - Violence Detection System

AI-powered violence detection system with image/video upload and live webcam support.

## 📁 Project Structure

```
netra-ai/
├── app.py                 # Main Flask application
├── requirements.txt        # Python dependencies
├── templates/
│   ├── index.html         # Upload page
│   ├── webcam.html        # Live webcam page
│   └── video.html         # Video playback page
├── static/
│   ├── uploads/           # Uploaded files
│   └── results/           # Processed results
├── Procfile               # Cloud deployment
├── runtime.txt            # Python version
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```cmd
pip install -r requirements.txt
```

### 2. Run the App
```cmd
python app.py
```

### 3. Open Browser
```
http://localhost:5000
```

## 📤 Upload Images/Videos

### How to Use:
1. Go to http://localhost:5000
2. Click the upload area
3. Select an image (JPG, PNG) or video (MP4, AVI)
4. Click "Upload"
5. View detection results

### Features:
- ✅ Instant violence detection on images
- ✅ Video playback support
- ✅ Confidence score display
- ✅ Result image download

## 📹 Live Webcam

1. Click "Live Webcam" button
2. Click "Start Camera"
3. View real-time detection
4. Click "Stop Camera" to end

## ☁️ Cloud Deployment

### Render (Free):
1. Push to GitHub
2. Go to https://dashboard.render.com
3. New → Web Service
4. Connect repo
5. Build: `pip install -r requirements.txt`
6. Start: `gunicorn app:app`

### Railway:
```cmd
npm install -g @railway/cli
railway login
railway init
railway up
```

## ⚠️ Important Notes

### For Cloud:
- Uploaded files are temporary on free tiers
- For permanent storage, use cloud storage (AWS S3, etc.)

### For Real Violence Detection:
The current detection is a placeholder. For real detection, implement:
- **YOLO** object detection
- **TensorFlow/PyTorch** models
- **Pre-trained violence detection models**

## 📋 Requirements

```
Flask==3.0.0
opencv-python==4.8.1.78
numpy==1.26.2
gunicorn==21.2.0
Werkzeug==3.0.1
Pillow==10.1.0
```

## 🔗 ngrok for Public Access

```cmd
ngrok http 5000
```

Share the HTTPS URL to access from anywhere!

---

**© 2024 Netra AI - AI-Powered Security System**
=======
# NetraAI
>>>>>>> dbb5b3061bbe12d46e91cdfd73e68e2b48be6cdd

"""
Netra AI - Violence Detection System
Flask + OpenCV + YOLOv8 Application
"""

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from datetime import datetime
import os
import base64
import time
from functools import wraps

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# SMS Configuration (Twilio)
# Get these from https://console.twilio.com
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '')
ALERT_PHONE_NUMBER = os.environ.get('ALERT_PHONE_NUMBER', '')

# Rate limiting for SMS alerts - Disabled for immediate delivery
last_sms_time = 0
SMS_COOLDOWN = 0  # No cooldown - SMS sent immediately

# Detection Statistics
detection_stats = {
    'total_detections': 0,
    'high_threat': 0,
    'medium_threat': 0,
    'low_threat': 0,
    'sms_sent': 0,
    'whatsapp_sent': 0,
    'total_response_time': 0,  # in milliseconds
    'detections': []  # List of recent detections
}

def send_sms_alert(confidence=0, detected_objects=None):
    """Send SMS alert when violence is detected"""
    global last_sms_time, detection_stats
    
    # Start timing for response time
    alert_start_time = time.time()
    
    # Check cooldown (disabled)
    current_time = time.time()
    if SMS_COOLDOWN > 0 and current_time - last_sms_time < SMS_COOLDOWN:
        print(f"SMS cooldown active. Next alert in {SMS_COOLDOWN - (current_time - last_sms_time):.0f}s")
        return False, "Cooldown active"
    
    # Check credentials
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER or not ALERT_PHONE_NUMBER:
        print("SMS not configured. Set Twilio environment variables.")
        return False, "SMS not configured - check .env"
    
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Format confidence as percentage
        confidence_pct = confidence * 100 if confidence <= 1 else confidence
        
        # Determine threat level
        if confidence_pct >= 70:
            threat_level = "HIGH"
        elif confidence_pct >= 40:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        # Build message - simple format without emojis
        objects_str = ""
        if detected_objects:
            objects_str = f" | Detected: {', '.join(detected_objects[:3])}"
        
        message_body = f"VIOLENCE ALERT! Threat: {threat_level} | Confidence: {confidence_pct:.1f}%{objects_str} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Netra AI Security"
        
        # Send SMS
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        
        # Small delay to ensure message is processed
        time.sleep(1)
        sms_sent = True
        
        # Also send WhatsApp message
        whatsapp_sent = False
        try:
            whatsapp_from = f"whatsapp:{TWILIO_PHONE_NUMBER}"
            whatsapp_to = f"whatsapp:{ALERT_PHONE_NUMBER}"
            
            whatsapp_body = f"🛡️ *VIOLENCE ALERT*\n\n" \
                           f"⚠️ Threat Level: {threat_level}\n" \
                           f"📊 Confidence: {confidence_pct:.1f}%\n" \
                           f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" \
                           f"🔒 Netra AI Security System"
            
            whatsapp_message = client.messages.create(
                body=whatsapp_body,
                from_=whatsapp_from,
                to=whatsapp_to
            )
            whatsapp_sent = True
            print(f"✅ WhatsApp alert sent! SID: {whatsapp_message.sid}")
        except Exception as wa_err:
            print(f"WhatsApp not sent: {wa_err}")
        
        # Calculate response time in milliseconds
        response_time = (time.time() - alert_start_time) * 1000
        
        # Update statistics
        detection_stats['total_detections'] += 1
        detection_stats['total_response_time'] += response_time
        if threat_level == "HIGH":
            detection_stats['high_threat'] += 1
        elif threat_level == "MEDIUM":
            detection_stats['medium_threat'] += 1
        else:
            detection_stats['low_threat'] += 1
        if sms_sent:
            detection_stats['sms_sent'] += 1
        if whatsapp_sent:
            detection_stats['whatsapp_sent'] += 1
        
        # Add to recent detections (keep last 10)
        detection_stats['detections'].insert(0, {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'threat_level': threat_level,
            'confidence': confidence_pct,
            'response_time': response_time,
            'objects': detected_objects[:3] if detected_objects else []
        })
        if len(detection_stats['detections']) > 10:
            detection_stats['detections'] = detection_stats['detections'][:10]
        
        last_sms_time = current_time
        print(f"✅ SMS alert sent!")
        print(f"   Message SID: {message.sid}")
        print(f"   From: {TWILIO_PHONE_NUMBER}")
        print(f"   To: {ALERT_PHONE_NUMBER}")
        print(f"   Stats: {threat_level} threat, {response_time:.0f}ms response time")
        return True, f"Alert sent - {threat_level} ({confidence_pct:.1f}%)"
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

# YOLOv8 imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not installed. Run: pip install ultralytics")

# HMM-based violence detection imports
try:
    from hmm_violence_model import HMMBasedViolenceDetector, ViolenceDetectionPipeline
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: HMM module not available. Run: pip install hmmlearn scikit-learn")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'netra-ai-secret-key-2024')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# User database (demo - in production use hashed passwords and database)
USERS = {
    'admin': {
        'password': 'admin123',
        'role': 'Administrator',
        'initial': 'A'
    },
    'security': {
        'password': 'secure456',
        'role': 'Security Officer',
        'initial': 'S'
    },
    'user': {
        'password': 'user123',
        'role': 'User',
        'initial': 'U'
    }
}

# User session data storage (in production use database)
user_settings = {}

# Store alert phone numbers per user
alert_phone_numbers = {}

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Initialize YOLOv8 model
model = None
if YOLO_AVAILABLE:
    try:
        model = YOLO('yolov8n.pt')  # Small model - fast
        print("YOLOv8 model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLOv8: {e}")
        model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_violence(image):
    """
    Violence detection using YOLOv8 object detection.
    Detects persons and potential weapons.
    """
    height, width = image.shape[:2]
    result_info = {
        'violence_detected': False,
        'confidence': 0.0,
        'message': 'Analysis complete',
        'detected_objects': [],
        'persons': 0,
        'weapons': []
    }
    
    if model is None:
        print("Warning: YOLOv8 model not loaded, using fallback motion detection")
        # Fallback to basic detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Motion detection
        global prev_frame
        if 'prev_frame' not in globals():
            prev_frame = gray
        
        frame_delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_ratio = cv2.countNonZero(thresh) / (width * height)
        prev_frame = gray.copy()
        
        # Violence score
        violence_score = 0
        if motion_ratio > 0.05:
            violence_score += 0.5
        
        result_info['violence_detected'] = violence_score > 0.3
        result_info['confidence'] = violence_score
        result_info['message'] = 'Basic motion detection (YOLOv8 not loaded)'
        return result_info
    
    # Use YOLOv8
    try:
        print(f"Running YOLOv8 detection on image with shape: {image.shape}")
        results = model(image, stream=False, verbose=False)
        
        weapon_classes = ['knife', 'gun', 'pistol', 'firearm', 'scissors']
        violence_score = 0
        detected_objects = []
        persons = 0
        weapons = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls].lower()
                
                detected_objects.append(f"{name}: {conf:.1%}")
                
                if name == 'person':
                    persons += 1
                    violence_score += conf * 0.5
                
                # Check for weapons
                for weapon in weapon_classes:
                    if weapon in name:
                        weapons.append(name)
                        violence_score += conf * 1.0
        
        result_info['detected_objects'] = detected_objects
        result_info['persons'] = persons
        result_info['weapons'] = weapons
        result_info['violence_detected'] = violence_score > 0.5
        result_info['confidence'] = min(violence_score, 1.0)
        
        if persons > 0:
            result_info['message'] = f"{persons} person(s) detected"
        if weapons:
            result_info['message'] += f", Weapon(s): {', '.join(weapons)}"
        if not detected_objects:
            result_info['message'] = "No objects detected"
            
    except Exception as e:
        print(f"Error in YOLOv8 detection: {str(e)}")
        result_info['message'] = f"Detection error: {str(e)}"
    
    return result_info

# Initialize HMM-based violence detector
hmm_detector = None
if YOLO_AVAILABLE and HMM_AVAILABLE:
    try:
        trained_model_path = 'runs/detect/violence_detection/weights/best.pt'
        if os.path.exists(trained_model_path):
            hmm_detector = HMMBasedViolenceDetector()
            # Try to load pre-trained HMM model
            if os.path.exists('hmm_violence_model.pkl'):
                hmm_detector.load_model('hmm_violence_model.pkl')
                print("HMM violence detection model loaded!")
            else:
                print("HMM model not found. Run training with --train-hmm flag.")
        else:
            print("Trained YOLO model not found. HMM requires trained YOLO first.")
    except Exception as e:
        print(f"Error initializing HMM: {e}")

def detect_violence_hmm(frame_sequence):
    """
    Violence detection using HMM for temporal sequence analysis.
    
    Args:
        frame_sequence: List of YOLO prediction results from consecutive frames
        
    Returns:
        Dictionary with violence detection results
    """
    result_info = {
        'violence_detected': False,
        'confidence': 0.0,
        'message': 'Analysis complete',
        'hmm_analysis': True,
        'temporal_pattern': 'N/A'
    }
    
    if hmm_detector is None or not hmm_detector.is_fitted:
        result_info['message'] = 'HMM model not available'
        result_info['hmm_available'] = False
        return result_info
    
    if len(frame_sequence) < 3:
        result_info['message'] = 'Insufficient frames for HMM analysis'
        result_info['hmm_available'] = True
        return result_info
    
    try:
        # Use HMM for prediction
        prediction, probabilities = hmm_detector.predict_sequence(
            frame_sequence, model
        )
        
        result_info['violence_detected'] = bool(prediction == 1)
        result_info['confidence'] = probabilities['violent']
        result_info['hmm_available'] = True
        
        # Analyze temporal pattern
        if prediction == 1:
            if probabilities['violent'] > 0.8:
                result_info['temporal_pattern'] = 'high_confidence_violent'
                result_info['message'] = 'Strong violent pattern detected'
            else:
                result_info['temporal_pattern'] = 'emerging_violence'
                result_info['message'] = 'Emerging violent pattern detected'
        else:
            if probabilities['peaceful'] > 0.8:
                result_info['temporal_pattern'] = 'high_confidence_peaceful'
                result_info['message'] = 'Consistent peaceful behavior'
            else:
                result_info['temporal_pattern'] = 'uncertain'
                result_info['message'] = 'Uncertain pattern - monitoring'
                
    except Exception as e:
        print(f"Error in HMM detection: {str(e)}")
        result_info['message'] = f"HMM error: {str(e)}"
        result_info['hmm_available'] = True
    
    return result_info

def process_video_hmm(filepath, frame_skip=3):
    """
    Process video using HMM for temporal violence detection.
    
    Args:
        filepath: Path to video file
        frame_skip: Process every Nth frame
        
    Returns:
        result_info: Violence detection results
    """
    result_info = {
        'violence_detected': False,
        'confidence': 0.0,
        'message': 'Video analysis complete',
        'hmm_analysis': True,
        'frames_analyzed': 0,
        'violent_frames': 0
    }
    
    if model is None:
        result_info['message'] = 'YOLO model not available'
        return result_info
    
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        result_info['message'] = 'Could not open video'
        return result_info
    
    frame_results = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            # Run YOLO detection on frame
            results = model(frame, stream=False, verbose=False)
            if results:
                frame_results.append(results[0])
        
        frame_count += 1
        
        # Limit frames for performance
        if frame_count > 150:
            break
    
    cap.release()
    
    result_info['frames_analyzed'] = len(frame_results)
    
    if len(frame_results) < 3:
        result_info['message'] = 'Insufficient frames for analysis'
        # Fall back to per-frame analysis
        for r in frame_results:
            boxes = r.boxes
            if len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls].lower()
                    if name == 'violent':
                        result_info['violent_frames'] += 1
        
        if result_info['violent_frames'] > len(frame_results) * 0.3:
            result_info['violence_detected'] = True
            result_info['confidence'] = result_info['violent_frames'] / len(frame_results)
        
        return result_info
    
    # Use HMM analysis
    hmm_result = detect_violence_hmm(frame_results)
    
    result_info['violence_detected'] = hmm_result['violence_detected']
    result_info['confidence'] = hmm_result['confidence']
    result_info['message'] = hmm_result['message']
    result_info['temporal_pattern'] = hmm_result.get('temporal_pattern', 'N/A')
    
    return result_info

def process_image(filepath):
    """Process uploaded image and detect violence"""
    image = cv2.imread(filepath)
    if image is None:
        return None, "Could not read image"
    
    # Resize for display
    max_width = 800
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, None, fx=scale, fy=scale)
    
    # Detect violence
    result = detect_violence(image.copy())
    
    # Draw results on image
    if result['violence_detected']:
        cv2.putText(image, "⚠️ VIOLENCE DETECTED", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(image, f"Confidence: {result['confidence']:.1%}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Send SMS alert - PHOTO UPLOAD
        print("\n" + "="*50)
        print("🚨 VIOLENCE DETECTED IN PHOTO - SENDING SMS")
        print("="*50)
        sms_success, sms_msg = send_sms_alert(result['confidence'], result.get('detected_objects', []))
        print(f"SMS Result: success={sms_success}, message={sms_msg}")
        print("="*50 + "\n")
    else:
        cv2.putText(image, "✅ NO VIOLENCE DETECTED", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(image, f"Confidence: {result['confidence']:.1%}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add detected objects info
    if result['detected_objects']:
        y_offset = 100
        for obj in result['detected_objects'][:5]:  # Show max 5
            cv2.putText(image, obj, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, f"Processed: {timestamp}", (10, image.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save result
    result_path = f"static/results/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(result_path, image)
    
    return result_path, result

# ============== Authentication Routes ==============

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Check credentials
        if username in USERS and USERS[username]['password'] == password:
            session['username'] = username
            session['role'] = USERS[username]['role']
            session['initial'] = USERS[username]['initial']
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password', demo_credentials=USERS)
    
    return render_template('login.html', demo_credentials=USERS)

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard page"""
    username = session.get('username', 'User')
    role = session.get('role', 'User')
    initial = session.get('initial', 'U')
    
    # Get user alert settings - use .env phone as default
    user_phone = alert_phone_numbers.get(username, ALERT_PHONE_NUMBER or '')
    
    # Format phone for display
    if user_phone:
        phone_display = user_phone
    else:
        phone_display = 'Not configured'
    
    return render_template('dashboard.html', 
                          username=username,
                          user_role=role,
                          user_initial=initial,
                          alert_phone=user_phone,
                          alert_phone_display=phone_display,
                          sms_enabled=True)

@app.route('/dashboard/upload')
@login_required
def dashboard_upload():
    """Dashboard upload page"""
    return redirect(url_for('index'))

@app.route('/dashboard/cameras')
@login_required
def dashboard_cameras():
    """Dashboard cameras page"""
    return redirect(url_for('webcam'))

@app.route('/dashboard/alerts')
@login_required
def dashboard_alerts():
    """Dashboard alerts page"""
    username = session.get('username', 'User')
    user_phone = alert_phone_numbers.get(username, '')
    return render_template('dashboard.html',
                          username=username,
                          user_role=session.get('role', 'User'),
                          user_initial=session.get('initial', 'U'),
                          alert_phone=user_phone,
                          alert_phone_display=user_phone or 'Not configured',
                          sms_enabled=True)

@app.route('/dashboard/settings')
@login_required
def dashboard_settings():
    """Dashboard settings page"""
    username = session.get('username', 'User')
    user_phone = alert_phone_numbers.get(username, '')
    return render_template('dashboard.html',
                          username=username,
                          user_role=session.get('role', 'User'),
                          user_initial=session.get('initial', 'U'),
                          alert_phone=user_phone,
                          alert_phone_display=user_phone or 'Not configured',
                          sms_enabled=True)

@app.route('/api/save-alert-settings', methods=['POST'])
@login_required
def save_alert_settings():
    """Save alert settings"""
    try:
        data = request.get_json()
        username = session.get('username', '')
        
        if username:
            alert_phone_numbers[username] = data.get('phone', '')
            user_settings[username] = {
                'sms_enabled': data.get('sms_enabled', True),
                'email_enabled': data.get('email_enabled', True)
            }
            
            # Update global ALERT_PHONE_NUMBER for SMS alerts
            global ALERT_PHONE_NUMBER
            ALERT_PHONE_NUMBER = data.get('phone', '')
            
            return jsonify({'success': True, 'message': 'Settings saved'})
        else:
            return jsonify({'success': False, 'error': 'User not authenticated'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test-sms', methods=['POST'])
@login_required
def test_sms():
    """Test SMS sending - sends a test message to verify Twilio setup"""
    try:
        print("\n" + "="*50)
        print("TEST SMS ENDPOINT CALLED")
        print("="*50)
        print(f"TWILIO_ACCOUNT_SID: {TWILIO_ACCOUNT_SID[:20]}..." if TWILIO_ACCOUNT_SID else "TWILIO_ACCOUNT_SID: Not set")
        print(f"TWILIO_AUTH_TOKEN: {'Set' if TWILIO_AUTH_TOKEN else 'Not set'}")
        print(f"TWILIO_PHONE_NUMBER: {TWILIO_PHONE_NUMBER}" if TWILIO_PHONE_NUMBER else "TWILIO_PHONE_NUMBER: Not set")
        print(f"ALERT_PHONE_NUMBER: {ALERT_PHONE_NUMBER}" if ALERT_PHONE_NUMBER else "ALERT_PHONE_NUMBER: Not set")
        print("="*50 + "\n")
        
        # Send test SMS with simulated values
        success, msg = send_sms_alert(0.85, ['person', 'car'])
        
        return jsonify({
            'success': success,
            'message': msg,
            'twilio_configured': bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER and ALERT_PHONE_NUMBER),
            'alert_phone': ALERT_PHONE_NUMBER
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test-whatsapp', methods=['POST'])
@login_required
def test_whatsapp():
    """Test WhatsApp message sending"""
    try:
        username = session.get('username', 'User')
        user_phone = alert_phone_numbers.get(username, ALERT_PHONE_NUMBER)
        
        print("\n" + "="*50)
        print("TEST WHATSAPP ENDPOINT CALLED")
        print("="*50)
        print(f"User: {username}")
        print(f"Phone: {user_phone}")
        print("="*50 + "\n")
        
        if not user_phone:
            return jsonify({'success': False, 'message': 'No phone number configured'})
        
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
            return jsonify({'success': False, 'message': 'Twilio not configured'})
        
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # WhatsApp messages use the format: whatsapp:+phonenumber
        # From number must be in format: whatsapp:+TWILIO_PHONE_NUMBER
        whatsapp_from = f"whatsapp:{TWILIO_PHONE_NUMBER}"
        whatsapp_to = f"whatsapp:{user_phone}"
        
        message_body = f"🛡️ *Netra AI Test Message*\n\nThis is a test WhatsApp alert from your violence detection system.\n\nIf you received this message, your WhatsApp alerts are configured correctly!"
        
        message = client.messages.create(
            body=message_body,
            from_=whatsapp_from,
            to=whatsapp_to
        )
        
        print(f"✅ WhatsApp message sent!")
        print(f"   Message SID: {message.sid}")
        
        return jsonify({
            'success': True,
            'message': 'WhatsApp test message sent!',
            'alert_phone': user_phone
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats():
    """Get detection statistics"""
    try:
        stats = detection_stats.copy()
        
        # Calculate average response time
        if stats['total_detections'] > 0:
            stats['avg_response_time'] = stats['total_response_time'] / stats['total_detections']
            # Get latest detection
            if stats['detections']:
                stats['latest_detection'] = stats['detections'][0]
        else:
            stats['avg_response_time'] = 0
            stats['latest_detection'] = None
        
        # Calculate percentages for threat levels
        total = stats['total_detections']
        if total > 0:
            stats['high_threat_pct'] = round((stats['high_threat'] / total) * 100, 1)
            stats['medium_threat_pct'] = round((stats['medium_threat'] / total) * 100, 1)
            stats['low_threat_pct'] = round((stats['low_threat'] / total) * 100, 1)
        else:
            stats['high_threat_pct'] = 0
            stats['medium_threat_pct'] = 0
            stats['low_threat_pct'] = 0
        
        # Remove internal data
        del stats['total_response_time']
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ============== Main Routes ==============

@app.route('/')
def index():
    """Home page - shows login page first"""
    # If user is logged in, redirect to dashboard
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html', demo_credentials=USERS)

@app.route('/api/trigger-alert', methods=['POST'])
@login_required
def trigger_alert():
    """Trigger SMS alert when violence is detected"""
    try:
        data = request.get_json()
        confidence = data.get('confidence', 0.5)
        objects = data.get('objects', [])
        
        print(f"\n🚨 VIOLENCE DETECTED - Triggering Alert!")
        print(f"Confidence: {confidence * 100:.1f}%")
        print(f"Objects: {objects}")
        print(f"Alert Phone from .env: {ALERT_PHONE_NUMBER}")
        
        success, msg = send_sms_alert(confidence, objects)
        
        if success:
            print(f"✅ SMS SENT SUCCESSFULLY!")
        else:
            print(f"❌ SMS FAILED: {msg}")
        
        return jsonify({
            'success': success,
            'message': msg
        })
    except Exception as e:
        print(f"Error triggering alert: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if filename.lower().endswith(('.mp4', '.avi', '.mkv')):
            return redirect(url_for('video_feed', filename=filename))
        else:
            result_path, result = process_image(filepath)
            if result_path:
                return jsonify({
                    'success': True,
                    'result_image': result_path,
                    'detection': result
                })
            else:
                return jsonify({'error': result})
    
    return jsonify({'error': 'Invalid file'})

@app.route('/webcam')
@login_required
def webcam():
    """Live webcam page"""
    return render_template('webcam.html')

@app.route('/webcam_feed')
@login_required
def webcam_feed():
    """Generate webcam frames with detection"""
    def generate():
        camera = cv2.VideoCapture(0)
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Detect violence
            result = detect_violence(frame.copy())
            
            # Add overlay
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Netra AI - {timestamp}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if result['violence_detected']:
                cv2.putText(frame, "⚠️ VIOLENCE DETECTED", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, f"Confidence: {result['confidence']:.1%}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Send SMS alert with confidence
                sms_success, sms_msg = send_sms_alert(result['confidence'], result.get('detected_objects', []))
                print(f"SMS Alert: {sms_msg}")
            else:
                cv2.putText(frame, f"Detected: {result['message']}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        camera.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/<filename>')
@login_required
def video_feed(filename):
    """Video streaming route"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('video.html', filename=filename, filepath=filepath)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for base64 image detection"""
    if 'image' in request.json:
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is not None:
            result = detect_violence(image)
            return jsonify(result)
    
    return jsonify({'error': 'Invalid image data'})

@app.route('/detect', methods=['POST'])
@login_required
def detect():
    """API endpoint for file upload detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'violence_detected': False, 'confidence': 0.0, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'violence_detected': False, 'confidence': 0.0, 'error': 'No file selected'})
        
        # Read image
        import io
        file_bytes = np.frombuffer(file.read(), np.uint8)
        
        if len(file_bytes) == 0:
            return jsonify({'violence_detected': False, 'confidence': 0.0, 'error': 'Empty file uploaded'})
        
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'violence_detected': False, 'confidence': 0.0, 'error': 'Could not decode image. Please upload a valid image file (jpg, png, etc.)'})
        
        print(f"Image loaded successfully: shape={image.shape}")
        
        result = detect_violence(image)
        print(f"Detection result: {result}")
        
        # Send SMS alert if violence detected
        if result.get('violence_detected'):
            print("Violence detected, sending SMS alert...")
            sms_sent = send_sms_alert()
            if sms_sent:
                print("SMS alert sent successfully!")
            else:
                print("SMS alert not sent (cooldown or not configured)")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in /detect endpoint: {str(e)}")
        return jsonify({'violence_detected': False, 'confidence': 0.0, 'error': f'Server error: {str(e)}'})

@app.route('/train-hmm', methods=['POST'])
def train_hmm():
    """Train HMM model with video data"""
    try:
        if not HMM_AVAILABLE:
            return jsonify({'success': False, 'error': 'HMM module not available'})
        
        if 'violent_dir' not in request.json or 'peaceful_dir' not in request.json:
            return jsonify({'success': False, 'error': 'Missing training directories'})
        
        violent_dir = request.json['violent_dir']
        peaceful_dir = request.json['peaceful_dir']
        
        # Initialize pipeline
        trained_model_path = 'runs/detect/violence_detection/weights/best.pt'
        if not os.path.exists(trained_model_path):
            return jsonify({'success': False, 'error': 'Trained YOLO model not found'})
        
        pipeline = ViolenceDetectionPipeline(trained_model_path)
        
        # Train HMM
        pipeline.train_hmm(training_data_dir='')
        
        # Reinitialize detector with trained model
        global hmm_detector
        hmm_detector = HMMBasedViolenceDetector()
        if os.path.exists('hmm_violence_model.pkl'):
            hmm_detector.load_model('hmm_violence_model.pkl')
            
        return jsonify({'success': True, 'message': 'HMM model trained successfully'})
        
    except Exception as e:
        print(f"Error training HMM: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/hmm-status')
def hmm_status():
    """Check HMM model status"""
    return jsonify({
        'hmm_available': HMM_AVAILABLE,
        'hmm_fitted': hmm_detector.is_fitted if hmm_detector else False,
        'yolo_available': YOLO_AVAILABLE
    })

@app.route('/detect-video', methods=['POST'])
def detect_video():
    """API endpoint for video violence detection using HMM"""
    try:
        if 'file' not in request.files:
            return jsonify({'violence_detected': False, 'confidence': 0.0, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'violence_detected': False, 'confidence': 0.0, 'error': 'No file selected'})
        
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mkv')):
            return jsonify({'violence_detected': False, 'confidence': 0.0, 'error': 'Invalid video format'})
        
        # Save uploaded video
        filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process with HMM
        result = process_video_hmm(filepath)
        
        # Clean up temp file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Send SMS alert if violence detected
        if result.get('violence_detected'):
            confidence = result.get('confidence', 0.5)
            objects = result.get('detected_objects', [])
            sms_success, sms_msg = send_sms_alert(confidence, objects)
            print(f"SMS Alert: {sms_msg}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in video detection: {str(e)}")
        return jsonify({'violence_detected': False, 'confidence': 0.0, 'error': str(e)})

if __name__ == '__main__':
    if not YOLO_AVAILABLE:
        print("Warning: Run 'pip install ultralytics' for YOLOv8 detection")
    app.run(host='0.0.0.0', port=5000, debug=True)

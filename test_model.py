from flask import Flask, render_template, Response, request, jsonify
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import webbrowser
import threading
import time
import datetime

app = Flask(__name__)

# ============================================
# 1️⃣ Load model and label mapping
# ============================================
model = load_model('model.keras')
label_mapping = np.load('label_mapping.npy', allow_pickle=True).item()
labels = list(label_mapping.keys())

# ============================================
# 2️⃣ MediaPipe setup
# ============================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ============================================
# 3️⃣ Globals
# ============================================
cap = None
last_prediction = None
sentence = ""
mode = "sign"

# ============================================
# 4️⃣ Feature extraction (same as training)
# ============================================
def calculate_distances(landmarks):
    distances = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            distances.append(dist)
    return np.array(distances)

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = np.array([[pt.x, pt.y, pt.z] for pt in hand_landmarks.landmark])
            distances = calculate_distances(lm)
            if len(distances) == 210:
                return distances.reshape(1, -1)
    return None

# ============================================
# 5️⃣ Frame generator
# ============================================
def generate_frames():
    global cap, last_prediction, sentence, mode

    last_char_time = datetime.datetime.now()

    while cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        data = preprocess_frame(frame)
        predicted_sign = ''
        confidence = 0.0

        if data is not None:
            prediction = model.predict(data, verbose=0)
            confidence = float(np.max(prediction))

            if confidence > 0.75:
                predicted_class = np.argmax(prediction, axis=1)[0]
                predicted_sign = labels[predicted_class]

                if mode == 'sign':
                    last_prediction = predicted_sign

                elif mode == 'sentence':
                    current_time = datetime.datetime.now()
                    time_diff = (current_time - last_char_time).total_seconds()

                    # Add letter if new or 1s delay
                    if predicted_sign != last_prediction or time_diff > 1.0:
                        last_prediction = predicted_sign
                        sentence += predicted_sign
                        last_char_time = current_time

        # Draw results
        if mode == 'sign' and last_prediction:
            cv2.putText(frame, f'Sign: {last_prediction}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif mode == 'sentence':
            cv2.putText(frame, f'Sentence: {sentence}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if confidence > 0:
            cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ============================================
# 6️⃣ Flask routes
# ============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_camera():
    global cap
    if not cap or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return jsonify({'status': 'camera started'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<string:selected_mode>', methods=['POST'])
def set_mode(selected_mode):
    global mode, sentence, last_prediction
    mode = selected_mode
    sentence = ""
    last_prediction = None
    return jsonify({'mode': mode})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global sentence
    sentence = ""
    return jsonify({'status': 'cleared'})

@app.route('/delete_last_letter', methods=['POST'])
def delete_last_letter():
    global sentence
    if len(sentence) > 0:
        sentence = sentence[:-1]
    return jsonify({'sentence': sentence})

@app.route('/add_space', methods=['POST'])
def add_space():
    global sentence
    sentence += ' '
    return jsonify({'sentence': sentence})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global cap
    if cap:
        cap.release()
    return jsonify({'status': 'shutting down'})

# ============================================
# 7️⃣ Launch
# ============================================
def open_browser():
    time.sleep(1)
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    try:
        threading.Thread(target=open_browser).start()
        app.run(debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("Shutting down...")
        if cap:
            cap.release()

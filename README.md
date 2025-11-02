# 🤟 Real-Time Sign Language Detection and Translation System.  

> A real-time **American Sign Language (ASL)** recognition web app that captures hand gestures using a webcam and predicts ASL letters instantly in your browser.  

---

## 🌟 Overview  

This project combines **computer vision** and **deep learning** to translate ASL hand gestures into text in real-time.  

- 🧠 Built with **TensorFlow/Keras** for deep learning  
- ✋ Uses **MediaPipe** for hand landmark detection  
- 🌐 Powered by **Flask** for the web interface  
- ⚡ Runs directly in your browser — no extra software required  


---

## 🧠 Features  

✨ **Smart Sign Recognition** — Detects ASL letters and words in real time using your webcam.  
🤖 **Deep Learning Powered** — Trained on hand landmark distances with a TensorFlow model.  
✋ **MediaPipe Integration** — Tracks 21 key hand landmarks for accurate recognition.  
📝 **Dual Modes** — Switch between **single sign** and **sentence** prediction modes.  
🗑️ **Interactive Controls** — Easily **add spaces**, **delete letters**, or **clear text** with on-screen buttons.  
💾 **Persistent Model & Labels** — Saves trained model (`model.keras`) and label mapping for seamless reuse.  

---

## 🗂️ Project Structure  

```plaintext
asl-sign-detection/
│
├── app.py               # Flask web application
├── train_model.py       # Model training script
├── model.keras          # Saved trained TensorFlow model
├── label_mapping.npy    # Encoded label dictionary
├── sign_data.csv        # Dataset of hand landmarks
│
├── templates/           # HTML templates
│   └── index.html       # Front-end UI
│
├── static/              # Static files (CSS, JS, images)
│   └── signs.png        # Example image asset
│
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/asl-sign-detection.git
cd asl-sign-detection
```

2️⃣ Create a virtual environment (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate       # On Windows
# or
source venv/bin/activate    # On Mac/Linux
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Run the Flask app
```bash
python app.py
```

Then open your browser and visit:
👉 http://127.0.0.1:5000

## 🧩 Training the Model
1. If you want to train your own model:

2. Prepare your dataset sign_data.csv (with columns like Distance_0 ... Distance_209 and a label column Sign).

### Run:

```bash
python train_model.py
```

3. The script will:

- Train a neural network

- Display accuracy, confusion matrix, and classification report

### Save:

- model.keras

- label_mapping.npy

## 🖥️ How It Works
1. MediaPipe detects 21 hand landmarks from the webcam feed.

2. Feature extraction calculates pairwise distances between landmarks (210 features).

3. Neural network model (trained with TensorFlow) predicts which ASL letter the hand sign represents.

4. Flask updates the live camera feed and displays:

5. Current sign

6. Confidence level

7. Sentence being formed

## 🎮 App Controls
Action	Description

▶️ Start Camera	Begins webcam feed

🪄 Mode: Sign	Detects one sign at a time

✍️ Mode: Sentence	Builds continuous text from signs

␣ Add Space	Adds a space to the sentence

⌫ Delete Last	Removes last character

🧹 Clear Sentence	Clears full text

🔻 Shutdown	Stops camera and app

## 🧾 Dependencies
Listed in requirements.txt:

- Flask
- numpy
- opencv-python
- mediapipe
- tensorflow
- scikit-learn
- matplotlib
- seaborn

## 🧱 Tech Stack  

| Category          | Technologies Used                                    |
|--------------------|-----------------------------------------------------|
| 🎨 **Frontend**     | HTML5, CSS3, JavaScript *(Flask Templates)*        |
| 🧩 **Backend**      | Python (Flask)                                     |
| 🧠 **ML / AI**       | TensorFlow, Keras                                 |
| ✋ **Computer Vision** | OpenCV, MediaPipe                               |
| 📊 **Visualization**  | Matplotlib, Seaborn                              |


## 🚀 Future Improvements
✋ Add support for dynamic signs (words/sentences via video)

🔤 Expand dataset with more signs

🌐 Host on Render / Hugging Face Spaces

📱 Create a mobile-friendly interface


## 📄 License
This project is licensed under the MIT License — you’re free to use, modify, and distribute it with attribution.

## ❤️ Acknowledgements
- Google MediaPipe

- TensorFlow

- OpenCV

---

⭐ If you found this project helpful, give it a star on GitHub! ⭐


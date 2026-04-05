<img width="500" height="500" alt="Blue_Water_Systems_Company_Logo-removebg-preview" src="https://github.com/user-attachments/assets/50dd58a0-b65c-474d-9560-0acd4be65997" />
**Realify AI**

🚀 Overview
Realify AI is an advanced AI-powered system that detects whether an image is Real or AI-Generated.
It goes beyond simple classification by performing forensic analysis, identifying hidden artifacts, and providing explainable results to ensure trust in digital media.

❗ Problem Statement
With the rise of generative AI, highly realistic fake images are being widely shared online.
This creates serious issues like misinformation, scams, identity misuse, and digital fraud.

There is a need for a system that can:
Detect AI-generated images accurately
Identify hidden artifacts
Provide clear explanations

Our Solution
Realify uses deep learning + forensic analysis to:
Classify images as REAL or AI-GENERATED
Detect pixel-level artifacts and inconsistencies
Provide confidence scores and explanations
Work effectively on real-world images (compressed, blurred, filtered)


🔗 Project Links
🚀 REALIFY: https://realifyai.onrender.com
📓 Kaggle Training Notebook:https://www.kaggle.com/code/ashaaajha/notebook3155268a4a

🚀 Key Features
Multi-Layered Defense: Analyzes Pixels (AI), Physics (FFT Frequency), and History (Metadata).
Human-Centric Forensics: Optimized to detect "Synthetic Smoothing" in morphed portraits.
FFT Grid Analysis: Spots mathematical noise signatures invisible to the human eye.
Metadata Integrity Scan: Verifies digital "Birth Certificates" to detect software-generated media.
Explainable AI (XAI): Generates forensic reasoning for every detection.

📊 Technical Summary
Component	Specification
Dataset Pool	1,000,000+ Images Scanned (Hybrid: CIFAKE + GenImage + Deepfake Faces)
Training Set	150,000 Balanced Samples (75K Real / 75K Fake)
Architecture	MobileNetV2 (High-Efficiency Transfer Learning)
Resolution	128x128 Pixels (Targeted Facial Forensics)
Accuracy	96% (Training)
Safety Metric	95%+ Recall (Optimized to never miss a fake image)
Loss	0.13 (Clean Convergence)

🛠️ Tech Stack
Deep Learning: Python, TensorFlow, Keras, MobileNetV2.
Computer Vision: OpenCV, NumPy (FFT Analysis), PIL (Metadata Extraction).
Web Engine: Flask (Backend), HTML5/CSS3 (Glassmorphism UI), JavaScript.
Infrastructure: Kaggle P100 GPU (Training), Render (Deployment).

🛡️ Social Impact & Mission
Realify was built with a critical mission: To stop AI-based blackmailing and harassment.
Standard detectors are often "blind" to high-quality human portraits. Realify bridges this gap by providing victims with a Forensic Shield. It empowers individuals to defend their authenticity with verifiable technical proof, moving the needle from passive detection to active legal defense.

🛠️ Local Setup
Clone & Install:
git clone https://github.com/kalyanijha445/realify.git
pip install flask tensorflow pillow numpy opencv-python
Model Weights: Place model_weights.weights.h5 in the root directory.

Run:
python app.py
🤝 Team
Vorqiels
Track: problem stt 7 Image Classification and Artifact Identification
for AI-Generated Images

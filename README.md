# 🎓 FaceAttend AI: Face Recognition Based Attendance Monitoring System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%20%7C%20LBPH-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Live%20App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://face-recognition-attendance-yhkfvrc6oelnzatkpz4cl7.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Status](https://img.shields.io/badge/System%20Status-Active-brightgreen.svg)]()

> **FaceAttend AI** is an automated, contactless biometric attendance system powered by **Computer Vision (Haar Cascade Face Detection)** and **Machine Learning (LBPH Face Recognition)**. It features a modern dual-architecture: a **Live Web Application (Streamlit SaaS)** accessible from any browser or smartphone, and an **Enterprise Desktop GUI (Tkinter + OpenCV)** for local deployment.

---

## 🚀 Live Demo & 1-Click Access

### 🌐 1. Live Web Application (Cloud Hosted)
Access the live deployed system on Streamlit Cloud directly from your phone or PC browser:

https://face-recognition-attendance-yhkfvrc6nelnzatkpz4cl7.streamlit.app/

---

### 💻 2. One-Touch Desktop Launcher (Local)
Double-click the bundled launcher in the project root to run locally without typing terminal commands:
```
Launch_Attendance_System.bat
```

---

## 🎯 2-Minute Project Pitch (For External Examiner / Viva)

If you need to explain this project to an external examiner in simple words, follow this 4-point structure:

1. **The Problem:** Traditional manual attendance takes 10–15 minutes of class time and is vulnerable to proxy marking. Biometric fingerprint devices require physical contact, raising hygiene concerns and queuing bottlenecks.
2. **The Solution:** A non-intrusive, optical facial recognition system that authenticates registered students in real-time, marks attendance automatically with timestamps, and generates daily CSV reports.
3. **The Core Algorithm:**
   - **Detection:** Uses *Haar Feature-based Cascade Classifiers* (`haarcascade_frontalface_default.xml`) to localize frontal faces in live video frames.
   - **Recognition:** Uses *Local Binary Patterns Histograms (LBPH)*. It converts 3x3 pixel neighborhoods into binary patterns, constructs histogram descriptors, and computes Chi-Square distance against trained embeddings to identify individuals even under lighting variations.
4. **Dual Deployment:**
   - **Cloud Web Interface:** Built with Streamlit for cross-platform browser access.
   - **Desktop Native GUI:** Built with Python Tkinter with a modern dark/light layout for offline computer labs.

---

## 🏗️ System Architecture & Workflow

```text
               ┌────────────────────────────────────────────────────────┐
               │                Live Video / Camera Input                │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │           Face Detection (Haar Cascade)                │
               │   • Converts frame to grayscale                        │
               │   • Detects bounding box coordinates (x, y, w, h)      │
               │   • Suppresses background noise (minSize: 80x80)       │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │      Biometric Feature Extraction & Recognition        │
               │   • Computes 8-neighbor Local Binary Patterns (LBP)    │
               │   • Generates spatially divided facial histograms      │
               │   • Computes Chi-Square confidence distance score       │
               └───────────────────────────┬────────────────────────────┘
                                           │
                       ┌───────────────────┴───────────────────┐
                       │                                       │
            Confidence < 60 (MATCH)                 Confidence >= 60 (UNKNOWN)
                       │                                       │
                       ▼                                       ▼
    ┌─────────────────────────────────────┐         ┌─────────────────────────┐
    │  • Identity Verified (ID + Name)    │         │  • Unrecognized Person  │
    │  • Record to Attendance_DATE.csv    │         │  • Alert Banner Display │
    │  • Display on Real-time Live Stream │         │  • Prompt Enrollment    │
    └─────────────────────────────────────┘         └─────────────────────────┘
```

---

## ✨ Key Features & Capabilities

- **Contactless Automated Check-in:** Real-time multi-frame facial recognition with zero physical contact required.
- **Dynamic Attendance Analytics:** Live dashboard metrics tracking Total Enrolled Students, Checked-in Today, Turnout Rate (%), and Recognition Model Health.
- **Automated CSV Generation:** Creates date-stamped CSV files (`Attendance_DD-MM-YYYY.csv`) automatically, preventing data overwrites across days.
- **Interactive Student Enrollment:** Step-by-step facial capture (collects 100 training frames) with visual progress feedback.
- **Admin Security:** Password-protected model training and profile management to prevent unauthorized database modifications.
- **Duplicate Prevention:** Intelligently checks previous check-ins within the same session to flag duplicate attempts gracefully.
- **Export & Search:** Built-in dynamic student search directory and instant one-click CSV download for faculty administration.

---

## 🛠️ Technology Stack

| Category | Component | Purpose |
|---|---|---|
| **Core Language** | Python 3.10+ | Primary application logic & runtime |
| **Computer Vision** | OpenCV (`cv2`) | Camera capture, frame processing, Haar cascades |
| **Biometric ML** | OpenCV Contrib (`cv2.face`) | LBPH (Local Binary Patterns Histogram) Face Recognizer |
| **Web Dashboard** | Streamlit | Cloud-ready responsive UI with browser webcam integration |
| **Desktop GUI** | Tkinter & `ttk` (Clam Theme) | High-performance standalone desktop interface |
| **Data Engine** | Pandas & NumPy | Array operations, DataFrame parsing, and CSV serialization |
| **Image Pipeline** | Pillow (PIL) | Multi-format image handling and dataset grayscale conversion |

---

## 📁 Repository Directory Structure

```text
Face-Recognition-Based-Attendance-Monitoring-System-main/
│
├── app.py                              # Modern Streamlit cloud web application
├── main.py                             # Modernized Tkinter native desktop application
├── Launch_Attendance_System.bat        # 1-Click launcher script for Windows
├── requirements.txt                    # Python runtime package dependencies
├── packages.txt                        # Linux system libraries for cloud deployment
├── haarcascade_frontalface_default.xml # Pre-trained OpenCV Haar Cascade model
│
├── .streamlit/
│   └── config.toml                     # Streamlit theme configuration (Light SaaS)
│
├── StudentDetails/
│   └── StudentDetails.csv             # Registered students directory (ID, Name, Serial)
│
├── TrainingImage/                      # Captured dataset face samples (.jpg)
│
├── TrainingImageLabel/
│   ├── Trainner.yml                   # Trained LBPH facial embeddings model file
│   └── psd.txt                        # Admin security verification password
│
└── Attendance/
    └── Attendance_DD-MM-YYYY.csv      # Daily generated attendance record logs
```

---

## 🏃 How to Run Locally

### Prerequisites
Make sure you have **Python 3.10+** installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/BharatD4/face-recognition-attendance.git
cd face-recognition-attendance
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Application

#### Option A: Web App (Streamlit in Browser)
```bash
streamlit run app.py
```
*Opens automatically at `http://localhost:8501`.*

#### Option B: Native Desktop GUI
```bash
python main.py
```

---

## 🎓 Viva Questions & Answers (Cheat Sheet for Presentation)

**Q1: Why did you use LBPH instead of Deep Learning (like CNN / FaceNet)?**  
> *Answer:* LBPH is computationally lightweight and runs in real-time on standard CPU hardware without requiring expensive GPUs. It works effectively even with small training datasets (5–100 images per person) and is invariant to monotonic grayscale illumination changes.

**Q2: How does Haar Cascade detect faces?**  
> *Answer:* Haar Cascade uses edge, line, and four-rectangle features evaluated rapidly using *Integral Images*. It utilizes the *AdaBoost* learning algorithm to select the most significant visual features and arranges them in a cascade of stages to reject non-face regions quickly.

**Q3: How is duplicate attendance handled?**  
> *Answer:* When a recognized face is detected, the system reads today's CSV file (`Attendance_DD-MM-YYYY.csv`). If a record with that Student ID already exists for the day, the UI displays a `VERIFIED (ALREADY RECORDED TODAY)` badge, preventing inflated counts.

**Q4: Where are the trained facial patterns stored?**  
> *Answer:* The LBPH model serializes its computed histogram patterns and labels into `TrainingImageLabel/Trainner.yml`. The system only reads this lightweight `.yml` file at runtime, meaning face recognition is near-instantaneous.

**Q5: Can this system be deployed for a full university?**  
> *Answer:* Yes. The architecture is modular: the CSV file storage can easily be connected to an enterprise SQL database (PostgreSQL/MySQL) or Cloud Firebase, and the Streamlit frontend can scale via cloud containers.

---

## 👨‍💻 Author & Maintainer

**Bharat Dewangan**  
- **GitHub:** [@BharatD4](https://github.com/BharatD4)  
- **Project Repository:** [face-recognition-attendance](https://github.com/BharatD4/face-recognition-attendance)

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).

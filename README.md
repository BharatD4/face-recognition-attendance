# 🎓 Face Recognition Based Attendance Monitoring System

A smart attendance system using **OpenCV LBPH Face Recognition** with a modern **Streamlit** web interface and original desktop GUI. Students can be registered via webcam, the model is trained on-the-fly, and attendance is recorded automatically by recognizing faces.

---

## 🚀 Live Demo & 1-Click Launch

### 🌐 Option 1: Live Cloud App (Direct Browser Link)
Click the badge or link below to open the live deployed attendance system directly:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://face-recognition-attendance-yhkfvrc6oelnzatkpz4cl7.streamlit.app)

👉 **[🔗 Click Here to Open Live Attendance System](https://face-recognition-attendance-yhkfvrc6oelnzatkpz4cl7.streamlit.app)**

---

### 💻 Option 2: 1-Click Desktop Launcher (Local)
Double-click `Launch_Attendance_System.bat` in the project folder to instantly start the application without typing commands.

---

## ✨ Key Features
- 📸 **Register Students:** Capture face photos directly via webcam
- 🤖 **Train Model:** Train OpenCV LBPH recognizer with a single click
- ✅ **Take Attendance:** Real-time face recognition and attendance logging
- 📊 **Attendance Records:** View and export daily attendance to CSV
- 🎨 **Modern UI:** Responsive dashboard with live statistics

---

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| **Face Recognition** | OpenCV (Haar Cascade + LBPH) |
| **Web Interface** | Streamlit |
| **Desktop GUI** | Tkinter |
| **Data Processing** | Pandas & NumPy |
| **Image Processing** | Pillow (PIL) |

---

## 🏃 Run Locally

### 1. Web App (Streamlit)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Desktop GUI (Tkinter)
```bash
pip install opencv-contrib-python pandas pillow
python main.py
```

---

## 📁 Project Structure
```
├── app.py                              # Streamlit web application
├── main.py                             # Tkinter desktop application
├── Launch_Attendance_System.bat        # 1-Click launcher
├── requirements.txt                    # Python dependencies
├── packages.txt                        # System dependencies for cloud
├── haarcascade_frontalface_default.xml # Haar Cascade face detector
├── StudentDetails/
│   └── StudentDetails.csv             # Registered students database
├── TrainingImage/                      # Captured face images
├── TrainingImageLabel/
│   └── Trainner.yml                   # Trained LBPH face model
└── Attendance/
    └── Attendance_DD-MM-YYYY.csv      # Daily attendance logs
```

---

## 👨‍💻 Author
**Bharat Dewangan**  
GitHub: [@BharatD4](https://github.com/BharatD4)

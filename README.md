# 🎓 Face Recognition Based Attendance Monitoring System

A smart attendance system using **OpenCV LBPH Face Recognition** with a modern **Streamlit** web interface. Students can be registered via webcam, the model is trained on-the-fly, and attendance is taken automatically by recognizing faces.

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## ✨ Features
- 📸 **Register Students** via browser webcam
- 🤖 **Train Model** automatically with OpenCV LBPH
- ✅ **Take Attendance** by face recognition in browser
- 📊 **View & Download** attendance records as CSV
- 🎨 **Modern Dark UI** with real-time stats dashboard

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| OpenCV (LBPH) | Face detection & recognition |
| Streamlit | Web UI |
| Pillow, NumPy | Image processing |
| Pandas | Attendance CSV management |
| Haar Cascade | Face detection |

## 🏃 Run Locally

### Web App (Streamlit)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Desktop GUI (Tkinter original)
```bash
pip install opencv-contrib-python pandas pillow
python main.py
```

## 📁 Project Structure
```
├── app.py                              ← Streamlit web app
├── main.py                             ← Original Tkinter desktop app
├── requirements.txt                    ← Python dependencies
├── haarcascade_frontalface_default.xml ← Face detector model
├── StudentDetails/
│   └── StudentDetails.csv             ← Registered students
├── TrainingImage/                      ← Face images for training
├── TrainingImageLabel/
│   └── Trainner.yml                   ← Trained LBPH model
└── Attendance/
    └── Attendance_DD-MM-YYYY.csv      ← Daily attendance files
```

## 🔄 How It Works
1. **Register** → Enter ID & Name → Capture 5 face photos via webcam
2. **Train** → Click "Save Profile" → LBPH model trains automatically
3. **Attend** → Go to "Take Attendance" → Look at camera → Marked ✅

## 👨‍💻 Author
Bharat Dewangan

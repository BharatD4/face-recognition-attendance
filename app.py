import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import csv
import datetime
import time

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION (Mobile & Desktop Responsive)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAttend AI | Smart Biometric Attendance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# PATHS & DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────
HAAR = "haarcascade_frontalface_default.xml"
STUDENT_CSV = os.path.join("StudentDetails", "StudentDetails.csv")
TRAINING_DIR = "TrainingImage"
LABEL_DIR = "TrainingImageLabel"
TRAINER_YML = os.path.join(LABEL_DIR, "Trainner.yml")
ATTENDANCE_DIR = "Attendance"

def assure_dirs():
    for d in [TRAINING_DIR, LABEL_DIR, ATTENDANCE_DIR, "StudentDetails"]:
        os.makedirs(d, exist_ok=True)

assure_dirs()

def detector():
    return cv2.CascadeClassifier(HAAR)

def load_students():
    if os.path.isfile(STUDENT_CSV):
        try:
            df = pd.read_csv(STUDENT_CSV)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
            df = df.loc[:, [c for c in df.columns if str(c).strip() != '']]
            df.columns = df.columns.str.strip()
            rename_map = {}
            for col in df.columns:
                c_upper = col.upper()
                if "SERIAL" in c_upper:
                    rename_map[col] = "Serial No"
                elif "NAME" in c_upper:
                    rename_map[col] = "Student Name"
                elif "ID" in c_upper:
                    rename_map[col] = "Student ID"
            df = df.rename(columns=rename_map)
            return df
        except Exception:
            return pd.DataFrame(columns=["Serial No", "Student ID", "Student Name"])
    return pd.DataFrame(columns=["Serial No", "Student ID", "Student Name"])

def total_registrations():
    return len(load_students())

def next_serial():
    return total_registrations() + 1

def save_student(serial, sid, name):
    exists = os.path.isfile(STUDENT_CSV)
    with open(STUDENT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["SERIAL NO.", "", "ID", "", "NAME"])
        writer.writerow([serial, "", sid, "", name])

def get_images_and_labels():
    img_paths = [
        os.path.join(TRAINING_DIR, f)
        for f in os.listdir(TRAINING_DIR)
        if f.lower().endswith(".jpg")
    ]
    faces, ids = [], []
    for path in img_paths:
        try:
            pil_img = Image.open(path).convert("L")
            img_np = np.array(pil_img, "uint8")
            parts = os.path.basename(path).split(".")
            id_val = int(parts[1])
            faces.append(img_np)
            ids.append(id_val)
        except Exception:
            pass
    return faces, ids

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels()
    if not faces:
        return False, 0
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_YML)
    return True, len(set(ids))

def recognize_primary_face(gray_img):
    if not os.path.isfile(TRAINER_YML):
        return None, []
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_YML)
    det = detector()
    
    # 80x80 minSize prevents tiny background noise from registering as false faces
    faces = det.detectMultiScale(gray_img, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return faces, []

    # Sort faces by area descending (largest foreground face evaluated first)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    df = load_students()

    results = []
    for (x, y, w, h) in faces[:2]:
        serial, conf = recognizer.predict(gray_img[y:y+h, x:x+w])
        if conf < 65:  # LBPH threshold
            match = df[df["Serial No"] == serial]
            if not match.empty:
                name = str(match.iloc[0]["Student Name"]).strip()
                sid = str(match.iloc[0]["Student ID"]).strip()
                match_pct = max(15, min(99, int(100 - conf)))
                results.append({"x": x, "y": y, "w": w, "h": h, "name": name, "id": sid, "confidence": match_pct, "status": "VERIFIED"})
                continue
        results.append({"x": x, "y": y, "w": w, "h": h, "name": "Unknown", "id": "N/A", "confidence": 0, "status": "UNRECOGNIZED"})

    return faces, results

def mark_attendance(sid, name):
    today = datetime.datetime.now().strftime("%d-%m-%Y")
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    file = os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")
    exists = os.path.isfile(file)
    already_logged = False

    if exists:
        try:
            curr_df = pd.read_csv(file)
            if not curr_df.empty:
                matches = curr_df[curr_df["ID"].astype(str) == str(sid)]
                if not matches.empty:
                    already_logged = True
        except Exception:
            pass

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["ID", "Name", "Date", "Time"])
        writer.writerow([sid, name, today, now_time])
        
    return already_logged, now_time

def load_attendance_today():
    today = datetime.datetime.now().strftime("%d-%m-%Y")
    file = os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")
    if os.path.isfile(file):
        try:
            df = pd.read_csv(file)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
            df = df.loc[:, [c for c in df.columns if str(c).strip() != '']]
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# RESPONSIVE MOBILE & DESKTOP CSS STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }
    
    /* App background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Mobile Container Optimization */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 1rem !important;
        }
    }

    /* Top Navigation Header */
    .saas-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        padding: 1.15rem 1.75rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon-box {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff;
        font-size: 19px;
        font-weight: 800;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
        flex-shrink: 0;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    .brand-desc {
        font-size: 0.82rem;
        color: #64748b;
        margin: 0;
        font-weight: 500;
    }
    .status-capsule {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #f1f5f9;
        padding: 7px 16px;
        border-radius: 30px;
        border: 1px solid #e2e8f0;
        font-size: 0.84rem;
        color: #334155;
        font-weight: 600;
        flex-shrink: 0;
    }
    .dot-live {
        width: 9px;
        height: 9px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
    }

    /* Mobile Header Stacking */
    @media (max-width: 640px) {
        .saas-header {
            flex-direction: column !important;
            align-items: flex-start !important;
            padding: 1rem !important;
            gap: 12px !important;
        }
        .status-capsule {
            width: 100% !important;
            justify-content: space-between !important;
            box-sizing: border-box !important;
        }
        .brand-title {
            font-size: 1.2rem !important;
        }
        .brand-desc {
            font-size: 0.75rem !important;
        }
    }

    /* Responsive KPI Grid Container (Desktop: 4 Col, Mobile/Tablet: 2 Col) */
    .kpi-grid-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 1.25rem;
    }
    @media (max-width: 900px) {
        .kpi-grid-container {
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
    }
    @media (max-width: 480px) {
        .kpi-grid-container {
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .card-kpi {
            padding: 0.9rem !important;
        }
        .card-kpi-val {
            font-size: 1.55rem !important;
        }
        .card-kpi-label {
            font-size: 0.68rem !important;
        }
        .card-kpi-sub {
            font-size: 0.72rem !important;
        }
    }

    /* Light KPI Cards */
    .card-kpi {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .card-kpi:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        border-color: #cbd5e1;
    }
    .card-kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .card-kpi-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    .card-kpi-sub {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
    }

    /* Light Theme Swipeable Tabs for Mobile */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f1f5f9;
        padding: 5px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        white-space: nowrap !important;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 18px;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.88rem;
        background: transparent;
        border: none;
        flex-shrink: 0 !important;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a;
        background-color: rgba(255, 255, 255, 0.6);
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        font-weight: 700;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.06) !important;
    }

    @media (max-width: 640px) {
        .stTabs [data-baseweb="tab"] {
            padding: 0 12px !important;
            font-size: 0.8rem !important;
            height: 38px !important;
        }
    }

    /* Mobile Camera Widget Styling */
    [data-testid="stCameraInput"] {
        width: 100% !important;
    }
    [data-testid="stCameraInput"] video {
        width: 100% !important;
        max-width: 100% !important;
        border-radius: 10px !important;
        object-fit: cover !important;
    }

    /* Dynamic Verification Banner */
    .banner-success-card {
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        border-radius: 12px;
        padding: 1.35rem;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.08);
    }
    .badge-success-pill {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 6px;
        background: #10b981;
        color: #ffffff;
        margin-bottom: 0.65rem;
    }
    .banner-warning-card {
        background: #fff1f2;
        border: 1.5px solid #fecdd3;
        border-radius: 12px;
        padding: 1.35rem;
        box-shadow: 0 2px 8px rgba(244, 63, 94, 0.08);
    }
    .badge-warning-pill {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 6px;
        background: #e11d48;
        color: #ffffff;
        margin-bottom: 0.65rem;
    }
    .student-title-lg {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }
    .meta-row-light {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        background: #ffffff;
        padding: 12px 14px;
        border-radius: 8px;
        border: 1px solid #dcfce7;
        margin-top: 1rem;
    }
    .meta-label-light {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        font-weight: 700;
    }
    .meta-val-light {
        font-size: 0.98rem;
        color: #0f172a;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    @media (max-width: 600px) {
        .meta-row-light {
            grid-template-columns: 1fr 1fr !important;
            gap: 10px !important;
            padding: 10px !important;
        }
        .student-title-lg {
            font-size: 1.35rem !important;
        }
    }

    /* Standby Light Card */
    .standby-card-light {
        border: 1.5px dashed #cbd5e1;
        background: #ffffff;
        border-radius: 12px;
        padding: 2.25rem 1.25rem;
        text-align: center;
        color: #64748b;
    }

    /* Section Typography */
    .section-headline {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .section-caption {
        font-size: 0.84rem;
        color: #64748b;
        margin-bottom: 1.25rem;
    }

    /* Primary Buttons & Mobile Touch Target */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        min-height: 44px !important;
        transition: transform 0.1s ease;
    }
    @media (max-width: 768px) {
        div.stButton > button {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOP LIGHT NAVBAR WITH REAL-TIME CLOCK & STATUS
# ─────────────────────────────────────────────────────────────
today_date_str = datetime.datetime.now().strftime("%d %B %Y")
now_time_str = datetime.datetime.now().strftime("%I:%M %p")

st.markdown(f"""
<div class="saas-header">
    <div class="brand-group">
        <div class="brand-icon-box">FA</div>
        <div>
            <h1 class="brand-title">FaceAttend AI</h1>
            <p class="brand-desc">Enterprise Biometric Attendance System</p>
        </div>
    </div>
    <div class="status-capsule">
        <span class="dot-live"></span>
        <span>Online</span>
        <span style="color:#cbd5e1;">|</span>
        <span style="font-family:'JetBrains Mono',monospace; color:#0f172a;">{today_date_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RESPONSIVE KPI BANNER (Desktop: 4-Col, Mobile: 2x2 Grid)
# ─────────────────────────────────────────────────────────────
total_enrolled = total_registrations()
today_attendance_df = load_attendance_today()
today_count = len(today_attendance_df)
attendance_rate_val = (today_count / total_enrolled * 100) if total_enrolled > 0 else 0
model_status_text = "LBPH Active" if os.path.isfile(TRAINER_YML) else "Pending"

st.markdown(f"""
<div class="kpi-grid-container">
    <div class="card-kpi">
        <div class="card-kpi-label">Total Students</div>
        <div class="card-kpi-val">{total_enrolled}</div>
        <div class="card-kpi-sub">Enrolled Profiles</div>
    </div>
    <div class="card-kpi">
        <div class="card-kpi-label">Checked-In Today</div>
        <div class="card-kpi-val" style="color:#2563eb;">{today_count}</div>
        <div class="card-kpi-sub">Verified Records</div>
    </div>
    <div class="card-kpi">
        <div class="card-kpi-label">Today's Turnout</div>
        <div class="card-kpi-val" style="color:#10b981;">{attendance_rate_val:.1f}%</div>
        <div class="card-kpi-sub">Attendance Rate</div>
    </div>
    <div class="card-kpi">
        <div class="card-kpi-label">Recognition Engine</div>
        <div class="card-kpi-val" style="font-size:1.35rem; color:#059669;">{model_status_text}</div>
        <div class="card-kpi-sub">Haar + LBPH Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS INTERFACE (Mobile Swipeable Segmented Control)
# ─────────────────────────────────────────────────────────────
tab_scanner, tab_enroll, tab_logs, tab_students = st.tabs([
    "📸 Live Attendance",
    "➕ Enroll Student",
    "📋 Today's Logs",
    "👥 Directory & History"
])

# ═════════════════════════════════════════════════════════════
# TAB 1: LIVE ATTENDANCE SCANNER
# ═════════════════════════════════════════════════════════════
with tab_scanner:
    if not os.path.isfile(TRAINER_YML) or not os.path.isfile(STUDENT_CSV):
        st.warning("⚠️ No facial biometric model found. Please enroll at least 1 student in the **Enroll Student** tab.")

    col_cam_box, col_verify_box = st.columns([1.1, 0.9], gap="medium")

    with col_cam_box:
        st.markdown('<div class="section-headline">Optical Biometric Scanner</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Align face within frame. Tap "Take photo" to verify.</div>', unsafe_allow_html=True)
        camera_frame = st.camera_input("Optical Scanner", label_visibility="collapsed")

    with col_verify_box:
        st.markdown('<div class="section-headline">Verification Result</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Real-time facial identity match and attendance record.</div>', unsafe_allow_html=True)

        if camera_frame is not None:
            raw_bytes = np.frombuffer(camera_frame.getvalue(), np.uint8)
            frame_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            faces_found, parsed_results = recognize_primary_face(gray)

            if len(faces_found) == 0:
                st.markdown("""
                <div class="banner-warning-card">
                    <span class="badge-warning-pill">NO FACE DETECTED</span>
                    <div class="student-title-lg" style="font-size:1.25rem;">Face Not Detected</div>
                    <div style="color:#64748b; font-size:0.85rem; margin-top:0.4rem;">
                        Please stand directly in front of the camera with adequate lighting and retake the photo.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Find the primary verified match
                verified_profile = next((r for r in parsed_results if r["status"] == "VERIFIED"), None)

                if verified_profile:
                    student_name = verified_profile["name"]
                    student_id = verified_profile["id"]
                    confidence_pct = verified_profile["confidence"]
                    already_marked, log_time = mark_attendance(student_id, student_name)

                    badge_text = "VERIFIED & LOGGED" if not already_marked else "VERIFIED (ALREADY RECORDED TODAY)"

                    st.markdown(f"""
                    <div class="banner-success-card">
                        <span class="badge-success-pill">{badge_text}</span>
                        <div class="student-title-lg">{student_name}</div>
                        <div class="meta-row-light">
                            <div>
                                <div class="meta-label-light">Student ID</div>
                                <div class="meta-val-light">{student_id}</div>
                            </div>
                            <div>
                                <div class="meta-label-light">Time Logged</div>
                                <div class="meta-val-light">{log_time}</div>
                            </div>
                            <div>
                                <div class="meta-label-light">Confidence</div>
                                <div class="meta-val-light" style="color:#10b981;">{confidence_pct}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.toast(f"✅ Attendance recorded for {student_name} ({student_id})")
                else:
                    st.markdown("""
                    <div class="banner-warning-card">
                        <span class="badge-warning-pill">UNRECOGNIZED IDENTITY</span>
                        <div class="student-title-lg" style="font-size:1.25rem;">Student Not In Database</div>
                        <div style="color:#64748b; font-size:0.85rem; margin-top:0.4rem;">
                            This face does not match any registered student profile. Please enroll via the Enroll Student tab.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="standby-card-light">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">📷</div>
                <div style="font-weight:700; color:#0f172a; font-size:0.95rem; margin-bottom:0.25rem;">Awaiting Shutter Capture</div>
                <div style="font-size:0.82rem; color:#64748b;">Tap "Take photo" on the camera feed to authenticate.</div>
            </div>
            """, unsafe_allow_html=True)

    # Real-Time Check-In Feed
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-headline">Real-Time Check-In Stream</div>', unsafe_allow_html=True)
    
    current_log_df = load_attendance_today()
    if not current_log_df.empty:
        st.dataframe(current_log_df.iloc[::-1], use_container_width=True, hide_index=True)
        csv_download_bytes = current_log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Today's Attendance (CSV)",
            data=csv_download_bytes,
            file_name=f"Attendance_{datetime.datetime.now().strftime('%d_%m_%Y')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No check-ins logged for today yet. When students scan their face, records will appear here dynamically.")

# ═════════════════════════════════════════════════════════════
# TAB 2: STUDENT ENROLLMENT (Registration)
# ═════════════════════════════════════════════════════════════
with tab_enroll:
    st.markdown('<div class="section-headline">Biometric Registration & Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Add new student profiles and train OpenCV facial recognition model.</div>', unsafe_allow_html=True)

    col_details, col_phototake = st.columns([1, 1], gap="medium")

    with col_details:
        st.markdown("##### 1. Student Information")
        new_sid = st.text_input("Student ID / Roll Number", placeholder="e.g. 101, CS-2024", key="form_sid")
        new_sname = st.text_input("Full Legal Name", placeholder="e.g. Bharat Dewangan", key="form_sname")
        target_sample_count = st.slider("Facial Training Samples", min_value=3, max_value=8, value=5, help="5 samples give the best balance of speed and accuracy.")

    with col_phototake:
        st.markdown("##### 2. Face Sample Capture")
        if not new_sid or not new_sname:
            st.info("Please fill in Student ID and Full Name on the left to activate face capture.")
        else:
            if "samples_taken" not in st.session_state:
                st.session_state.samples_taken = 0

            clean_name = "".join(c for c in new_sname if c.isalnum() or c in (" ", "_", "-")).strip()

            st.caption(f"Progress: {st.session_state.samples_taken} of {target_sample_count} photos captured")
            dyn_progress = min(1.0, st.session_state.samples_taken / target_sample_count)
            st.progress(dyn_progress)

            sample_capture = st.camera_input(f"Capture Sample #{st.session_state.samples_taken + 1}", key=f"enroll_cam_{st.session_state.samples_taken}")

            if sample_capture is not None:
                capture_bytes = np.frombuffer(sample_capture.getvalue(), np.uint8)
                frame_gray = cv2.imdecode(capture_bytes, cv2.IMREAD_GRAYSCALE)
                det = detector()
                faces_in_sample = det.detectMultiScale(frame_gray, 1.25, 5, minSize=(80, 80))

                if len(faces_in_sample) == 0:
                    st.error("Face not clearly detected. Ensure proper lighting and center your face.")
                else:
                    serial_id = next_serial()
                    faces_sorted = sorted(faces_in_sample, key=lambda f: f[2] * f[3], reverse=True)
                    x, y, w, h = faces_sorted[0]
                    sample_no = st.session_state.samples_taken + 1
                    file_name = f"{clean_name}.{serial_id}.{new_sid}.{sample_no}.jpg"
                    file_target = os.path.join(TRAINING_DIR, " " + file_name)
                    cv2.imwrite(file_target, frame_gray[y:y+h, x:x+w])

                    st.session_state.samples_taken += 1
                    st.success(f"Sample {sample_no} saved successfully.")

            if st.session_state.samples_taken >= target_sample_count:
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if st.button("💾 Enroll & Train Model", type="primary", use_container_width=True):
                        serial_id = next_serial()
                        save_student(serial_id, new_sid, new_sname)
                        with st.spinner("Training facial recognition network with new biometric data..."):
                            success, trained_total = train_model()
                        if success:
                            st.balloons()
                            st.success(f"Enrollment Complete! {new_sname} ({new_sid}) registered across {trained_total} biometric profiles.")
                            st.session_state.samples_taken = 0
                        else:
                            st.error("Model training failed. Please re-capture samples.")
                with btn_c2:
                    if st.button("🔄 Reset / Enroll Another", use_container_width=True):
                        st.session_state.samples_taken = 0
                        st.rerun()

# ═════════════════════════════════════════════════════════════
# TAB 3: TODAY'S ATTENDANCE LOGS
# ═════════════════════════════════════════════════════════════
with tab_logs:
    st.markdown('<div class="section-headline">Daily Attendance Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Comprehensive real-time view of all students marked present today.</div>', unsafe_allow_html=True)

    logs_df = load_attendance_today()
    if not logs_df.empty:
        st.markdown(f"**Total Records:** `{len(logs_df)} check-ins logged today`")
        csv_data = logs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Daily CSV",
            data=csv_data,
            file_name=f"Attendance_{datetime.datetime.now().strftime('%d_%m_%Y')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(logs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No attendance records logged for today yet.")

# ═════════════════════════════════════════════════════════════
# TAB 4: STUDENT DIRECTORY & HISTORICAL ARCHIVE
# ═════════════════════════════════════════════════════════════
with tab_students:
    st.markdown('<div class="section-headline">Enrolled Student Directory</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Master database of all registered students with instant dynamic search.</div>', unsafe_allow_html=True)

    master_df = load_students()
    if not master_df.empty:
        search_query = st.text_input("🔍 Dynamic Search (Search by Name or ID)", placeholder="Type student name or ID...", key="master_search")
        if search_query:
            matched_df = master_df[
                master_df["Student Name"].astype(str).str.contains(search_query, case=False, na=False) |
                master_df["Student ID"].astype(str).str.contains(search_query, case=False, na=False)
            ]
            st.dataframe(matched_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(master_df, use_container_width=True, hide_index=True)
    else:
        st.info("No enrolled students in the directory.")

    st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-headline">Historical Date Archive</div>', unsafe_allow_html=True)

    all_csv_history = sorted(
        [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")],
        reverse=True
    ) if os.path.isdir(ATTENDANCE_DIR) else []

    if all_csv_history:
        date_pick = st.selectbox("Select Past Attendance Date", all_csv_history)
        if date_pick:
            archive_df = pd.read_csv(os.path.join(ATTENDANCE_DIR, date_pick))
            archive_df = archive_df.loc[:, ~archive_df.columns.str.contains('^Unnamed', na=False)]
            archive_df = archive_df.loc[:, [c for c in archive_df.columns if str(c).strip() != '']]
            archive_df.columns = archive_df.columns.str.strip()
            st.dataframe(archive_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='height:35px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.75rem; border-top:1px solid #e2e8f0; padding-top:1.25rem;">
    FaceAttend AI • Enterprise Biometric Attendance Architecture • Powered by OpenCV LBPH & Streamlit
</div>
""", unsafe_allow_html=True)

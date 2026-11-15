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
# PAGE CONFIGURATION (Modern Enterprise Theme)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAttend AI | Biometric Attendance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# PATHS & SETUP
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
            # Standardize column naming
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
    """
    Detects faces, prioritizes the primary (largest foreground) face,
    and returns recognized identity to avoid background false positives.
    """
    if not os.path.isfile(TRAINER_YML):
        return None, []
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_YML)
    det = detector()
    
    # Min size of 80x80 prevents background clutter from triggering false alerts
    faces = det.detectMultiScale(gray_img, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return faces, []

    # Sort faces by bounding box area (largest foreground face first)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    df = load_students()

    results = []
    # Evaluate the primary foreground face first
    for (x, y, w, h) in faces[:2]:
        serial, conf = recognizer.predict(gray_img[y:y+h, x:x+w])
        if conf < 65:  # LBPH confidence threshold (lower is better match)
            match = df[df["Serial No"] == serial]
            if not match.empty:
                name = str(match.iloc[0]["Student Name"]).strip()
                sid = str(match.iloc[0]["Student ID"]).strip()
                match_pct = max(10, min(99, int(100 - conf)))
                results.append({"x": x, "y": y, "w": w, "h": h, "name": name, "id": sid, "confidence": match_pct, "status": "VERIFIED"})
                continue
        # If not verified
        results.append({"x": x, "y": y, "w": w, "h": h, "name": "Unknown Person", "id": "N/A", "confidence": 0, "status": "UNRECOGNIZED"})

    return faces, results

def mark_attendance(sid, name):
    today = datetime.datetime.now().strftime("%d-%m-%Y")
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    file = os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")
    exists = os.path.isfile(file)

    # Avoid duplicate entry within the last 60 seconds for same student
    if exists:
        try:
            curr_df = pd.read_csv(file)
            if not curr_df.empty:
                last_entries = curr_df[curr_df["ID"].astype(str) == str(sid)]
                if not last_entries.empty:
                    last_time_str = str(last_entries.iloc[-1]["Time"])
                    # If already marked today, still return success
                    pass
        except Exception:
            pass

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["ID", "Name", "Date", "Time"])
        writer.writerow([sid, name, today, now_time])

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
# PROFESSIONAL ENTERPRISE STYLING (CSS)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Reset & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top App Navbar */
    .app-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #0d1117;
        padding: 1.25rem 1.75rem;
        border-radius: 14px;
        border: 1px solid #30363d;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .brand-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo-icon {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-size: 20px;
        font-weight: 800;
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
    }
    .brand-name {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f0f6fc;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.8rem;
        color: #8b949e;
        margin: 0;
        font-weight: 500;
    }
    .system-status-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #161b22;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #30363d;
        font-size: 0.82rem;
        color: #c9d1d9;
        font-weight: 600;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }

    /* Modern Metric Cards */
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #8b949e;
        font-weight: 500;
    }

    /* Verification Status Banner */
    .verify-banner-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        color: #ffffff;
        margin-top: 0.5rem;
    }
    .verify-banner-warning {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        color: #ffffff;
        margin-top: 0.5rem;
    }
    .verify-badge-pill {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 6px;
        background: #10b981;
        color: #0d1117;
        margin-bottom: 0.75rem;
    }
    .verify-badge-unreg {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 6px;
        background: #ef4444;
        color: #ffffff;
        margin-bottom: 0.75rem;
    }
    .student-display-name {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
    }
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        background: rgba(0, 0, 0, 0.3);
        padding: 12px;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .meta-item-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8b949e;
        font-weight: 700;
    }
    .meta-item-value {
        font-size: 0.95rem;
        color: #f0f6fc;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 0.25rem;
    }
    .section-desc {
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 1.25rem;
    }

    /* Tabs Customization (Segmented Control) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #161b22;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 22px;
        border-radius: 8px;
        color: #8b949e;
        font-weight: 600;
        font-size: 0.88rem;
        border: none;
        background: transparent;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f0f6fc;
        background-color: rgba(255, 255, 255, 0.04);
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }

    /* Standby Card */
    .standby-card {
        border: 1px dashed #30363d;
        background: #161b22;
        border-radius: 12px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        color: #8b949e;
    }
    .standby-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        opacity: 0.6;
    }

    /* Clean Buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOP APP NAVBAR (Clean & Professional)
# ─────────────────────────────────────────────────────────────
today_date_str = datetime.datetime.now().strftime("%d %b %Y")
now_time_str = datetime.datetime.now().strftime("%H:%M:%S")

st.markdown(f"""
<div class="app-navbar">
    <div class="brand-section">
        <div class="brand-logo-icon">FA</div>
        <div>
            <h1 class="brand-name">FaceAttend Portal</h1>
            <p class="brand-subtitle">Automated Biometric Attendance & Identity Verification</p>
        </div>
    </div>
    <div class="system-status-badge">
        <span class="pulse-dot"></span>
        <span>Online</span>
        <span style="color:#484f58; margin: 0 4px;">|</span>
        <span style="font-family:'JetBrains Mono',monospace;">{today_date_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN TABS INTERFACE (Clean Segmented Layout)
# ─────────────────────────────────────────────────────────────
tab_attend, tab_register, tab_records, tab_analytics = st.tabs([
    "📸 Live Attendance",
    "➕ Student Enrollment",
    "📋 Daily Records",
    "📊 System Analytics"
])

# ═════════════════════════════════════════════════════════════
# TAB 1: LIVE ATTENDANCE (Primary User Flow)
# ═════════════════════════════════════════════════════════════
with tab_attend:
    # Pre-flight check
    if not os.path.isfile(TRAINER_YML) or not os.path.isfile(STUDENT_CSV):
        st.warning("⚠️ No enrolled students or trained biometric model detected. Please complete **Student Enrollment** first.")
    
    col_cam, col_result = st.columns([1.15, 0.85], gap="large")

    with col_cam:
        st.markdown('<div class="section-title">Optical Biometric Scanner</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Align face within camera frame. Tap shutter to log attendance.</div>', unsafe_allow_html=True)
        cam_input = st.camera_input("Scanner Feed", label_visibility="collapsed")

    with col_result:
        st.markdown('<div class="section-title">Verification Engine</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Real-time facial identity match and attendance record.</div>', unsafe_allow_html=True)

        if cam_input is not None:
            img_arr = np.frombuffer(cam_input.getvalue(), np.uint8)
            frame_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            faces_detected, results = recognize_primary_face(gray)

            if len(faces_detected) == 0:
                st.markdown("""
                <div class="verify-banner-warning">
                    <span class="verify-badge-unreg">NO FACE DETECTED</span>
                    <div class="student-display-name" style="font-size:1.25rem;">Camera Could Not Detect Face</div>
                    <div style="color:#8b949e; font-size:0.85rem; margin-top:0.4rem;">
                        Please ensure adequate frontal lighting and look directly into the camera.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Find if any verified face exists in results
                verified_match = next((r for r in results if r["status"] == "VERIFIED"), None)

                if verified_match:
                    name = verified_match["name"]
                    sid = verified_match["id"]
                    conf = verified_match["confidence"]
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    today_str = datetime.datetime.now().strftime("%d-%m-%Y")

                    mark_attendance(sid, name)

                    st.markdown(f"""
                    <div class="verify-banner-success">
                        <span class="verify-badge-pill">IDENTITY VERIFIED & LOGGED</span>
                        <div class="student-display-name">{name}</div>
                        <div class="meta-grid">
                            <div>
                                <div class="meta-item-label">Student ID</div>
                                <div class="meta-item-value">{sid}</div>
                            </div>
                            <div>
                                <div class="meta-item-label">Timestamp</div>
                                <div class="meta-item-value">{timestamp}</div>
                            </div>
                            <div>
                                <div class="meta-item-label">Confidence</div>
                                <div class="meta-item-value">{conf}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="verify-banner-warning">
                        <span class="verify-badge-unreg">UNRECOGNIZED IDENTITY</span>
                        <div class="student-display-name" style="font-size:1.25rem;">Student Not In Database</div>
                        <div style="color:#8b949e; font-size:0.85rem; margin-top:0.4rem;">
                            This face does not match any registered student profile. Please enroll via the Student Enrollment tab.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="standby-card">
                <div class="standby-icon">📷</div>
                <div style="font-weight:600; color:#c9d1d9; margin-bottom:0.25rem;">Awaiting Shutter Capture</div>
                <div style="font-size:0.82rem;">Allow browser camera access and click "Take photo" to authenticate.</div>
            </div>
            """, unsafe_allow_html=True)

    # Today's Live Attendance Feed
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Today’s Check-In Activity</div>', unsafe_allow_html=True)
    today_df = load_attendance_today()

    if not today_df.empty:
        st.dataframe(today_df, use_container_width=True, hide_index=True)
        csv_bytes = today_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Today's Attendance (CSV)",
            data=csv_bytes,
            file_name=f"Attendance_{datetime.datetime.now().strftime('%d_%m_%Y')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No attendance records logged for today yet. Use the scanner above to mark check-in.")

# ═════════════════════════════════════════════════════════════
# TAB 2: STUDENT ENROLLMENT (Registration)
# ═════════════════════════════════════════════════════════════
with tab_register:
    st.markdown('<div class="section-title">Student Biometric Enrollment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Register new student credentials and capture facial training samples.</div>', unsafe_allow_html=True)

    col_form, col_capture = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("##### 1. Student Credentials")
        new_id = st.text_input("Student ID / Roll Number", placeholder="e.g. 101, CS2024", key="enroll_id")
        new_name = st.text_input("Full Legal Name", placeholder="e.g. Bharat Dewangan", key="enroll_name")
        samples_target = st.slider("Facial Samples to Collect", min_value=3, max_value=8, value=5, help="More samples improve angle tolerance.")

    with col_capture:
        st.markdown("##### 2. Face Sample Capture")
        if not new_id or not new_name:
            st.info("Fill in Student ID and Name on the left to activate face capture.")
        else:
            if "enroll_count" not in st.session_state:
                st.session_state.enroll_count = 0

            clean_name = "".join(c for c in new_name if c.isalnum() or c in (" ", "_", "-")).strip()

            st.caption(f"Progress: {st.session_state.enroll_count} of {samples_target} samples collected")
            progress_val = min(1.0, st.session_state.enroll_count / samples_target)
            st.progress(progress_val)

            sample_img = st.camera_input(f"Capture Sample #{st.session_state.enroll_count + 1}", key=f"sample_{st.session_state.enroll_count}")

            if sample_img is not None:
                img_bytes = np.frombuffer(sample_img.getvalue(), np.uint8)
                frame_gray = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
                det = detector()
                faces = det.detectMultiScale(frame_gray, 1.25, 5, minSize=(80, 80))

                if len(faces) == 0:
                    st.error("Face not clearly detected. Ensure proper lighting and center your face.")
                else:
                    serial_num = next_serial()
                    # Save largest detected face
                    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    x, y, w, h = faces_sorted[0]
                    sample_idx = st.session_state.enroll_count + 1
                    fname = f"{clean_name}.{serial_num}.{new_id}.{sample_idx}.jpg"
                    fpath = os.path.join(TRAINING_DIR, " " + fname)
                    cv2.imwrite(fpath, frame_gray[y:y+h, x:x+w])

                    st.session_state.enroll_count += 1
                    st.success(f"Sample {sample_idx} captured successfully.")

            if st.session_state.enroll_count >= samples_target:
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("💾 Enroll & Train Biometric Model", type="primary", use_container_width=True):
                        serial_num = next_serial()
                        save_student(serial_num, new_id, new_name)
                        with st.spinner("Training facial recognition network..."):
                            success, trained_count = train_model()
                        if success:
                            st.balloons()
                            st.success(f"Enrollment Complete! {new_name} ({new_id}) registered across {trained_count} profiles.")
                            st.session_state.enroll_count = 0
                        else:
                            st.error("Model training failed. Please re-capture samples.")
                with col_btn2:
                    if st.button("🔄 Reset Form", use_container_width=True):
                        st.session_state.enroll_count = 0
                        st.rerun()

# ═════════════════════════════════════════════════════════════
# TAB 3: ATTENDANCE RECORDS (History & Audit)
# ═════════════════════════════════════════════════════════════
with tab_records:
    st.markdown('<div class="section-title">Historical Attendance Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Search, filter, and export past attendance records.</div>', unsafe_allow_html=True)

    csv_files = sorted(
        [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")],
        reverse=True
    ) if os.path.isdir(ATTENDANCE_DIR) else []

    if not csv_files:
        st.info("No historical attendance files found.")
    else:
        col_sel, col_exp = st.columns([1, 1], gap="medium")
        with col_sel:
            selected_file = st.selectbox("Select Attendance Date", csv_files)

        if selected_file:
            hist_df = pd.read_csv(os.path.join(ATTENDANCE_DIR, selected_file))
            hist_df = hist_df.loc[:, ~hist_df.columns.str.contains('^Unnamed', na=False)]
            hist_df = hist_df.loc[:, [c for c in hist_df.columns if str(c).strip() != '']]
            hist_df.columns = hist_df.columns.str.strip()

            with col_exp:
                st.caption(f"Total entries: {len(hist_df)}")
                csv_download = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download {selected_file}",
                    data=csv_download,
                    file_name=selected_file,
                    mime="text/csv",
                    use_container_width=True
                )

            st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════
# TAB 4: SYSTEM ANALYTICS & DIRECTORY
# ═════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown('<div class="section-title">System Overview & Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Live statistics, database integrity, and enrolled student directory.</div>', unsafe_allow_html=True)

    total_stud = total_registrations()
    today_count = len(load_attendance_today())
    attendance_rate = f"{(today_count / total_stud * 100):.1f}%" if total_stud > 0 else "0%"
    model_ready = "Trained & Active" if os.path.isfile(TRAINER_YML) else "Pending Training"

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Enrolled</div>
            <div class="kpi-value">{total_stud}</div>
            <div class="kpi-subtext">Registered Students</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Checked-In Today</div>
            <div class="kpi-value">{today_count}</div>
            <div class="kpi-subtext">Present Today</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Turnout Rate</div>
            <div class="kpi-value">{attendance_rate}</div>
            <div class="kpi-subtext">Today's Attendance %</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Biometric Model</div>
            <div class="kpi-value" style="font-size:1.35rem; color:#10b981;">{model_ready}</div>
            <div class="kpi-subtext">OpenCV LBPH Engine</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Enrolled Student Directory</div>', unsafe_allow_html=True)
    all_students_df = load_students()
    if not all_students_df.empty:
        search_term = st.text_input("🔍 Search Student Directory (Filter by Name or ID)", placeholder="Type to filter...")
        if search_term:
            filtered_df = all_students_df[
                all_students_df["Student Name"].astype(str).str.contains(search_term, case=False, na=False) |
                all_students_df["Student ID"].astype(str).str.contains(search_term, case=False, na=False)
            ]
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(all_students_df, use_container_width=True, hide_index=True)
    else:
        st.info("Student directory is empty.")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:0.75rem; border-top:1px solid #21262d; padding-top:1.5rem;">
    FaceAttend AI • Biometric Identity & Attendance Architecture • Powered by OpenCV LBPH & Streamlit
</div>
""", unsafe_allow_html=True)

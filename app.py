import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import csv
import datetime
import time

# ───────────────────────── PAGE CONFIG ─────────────────────────
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="🎓",
    layout="wide",
)

# ───────────────────────── HELPERS ─────────────────────────────
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
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            return pd.DataFrame(columns=["SERIAL NO.", "ID", "NAME"])
    return pd.DataFrame(columns=["SERIAL NO.", "ID", "NAME"])

def total_registrations():
    df = load_students()
    return len(df)

def next_serial():
    df = load_students()
    return len(df) + 1

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
            # filename pattern: " Name.serial.ID.samplenum.jpg"
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

def recognize_face(gray_img):
    if not os.path.isfile(TRAINER_YML):
        return None, None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_YML)
    det = detector()
    faces = det.detectMultiScale(gray_img, 1.2, 5)
    results = []
    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray_img[y:y+h, x:x+w])
        if conf < 50:
            df = load_students()
            df.columns = df.columns.str.strip()
            match = df[df["SERIAL NO."] == serial]
            if not match.empty:
                name = str(match.iloc[0]["NAME"]).strip()
                sid = str(match.iloc[0]["ID"]).strip()
                results.append((x, y, w, h, name, sid, conf))
        else:
            results.append((x, y, w, h, "Unknown", "-", conf))
    return faces, results

def mark_attendance(sid, name):
    today = datetime.datetime.now().strftime("%d-%m-%Y")
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    file = os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")
    exists = os.path.isfile(file)
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
            return pd.read_csv(file)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# ───────────────────────── CUSTOM CSS ──────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 10px rgba(233,69,96,0.5);
    }
    .main-header p {
        color: #a8b2d8;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .metric-card h2 { font-size: 2.5rem; margin: 0; font-weight: 800; }
    .metric-card p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        margin: 1rem 0;
    }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: transform 0.1s !important;
    }
    .stButton > button:hover { transform: scale(1.02); }
    section[data-testid="stSidebar"] { background: #0f3460 !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── HEADER ──────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Face Recognition Attendance System</h1>
    <p>AI-powered automatic attendance using OpenCV LBPH Face Recognition</p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────── SIDEBAR NAV ─────────────────────────
with st.sidebar:
    st.markdown("## 📋 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Dashboard", "📸 Register New Student", "✅ Take Attendance", "📊 Attendance Records"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"**Total Registered:** {total_registrations()}")
    st.markdown(f"**Date:** {datetime.datetime.now().strftime('%d %b %Y')}")
    st.markdown(f"**Time:** {datetime.datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")
    st.info("📌 Steps:\n1. Register student\n2. Train model\n3. Take attendance")

# ════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{total_registrations()}</h2>
            <p>👤 Total Registered Students</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        today_df = load_attendance_today()
        count = len(today_df) if not today_df.empty else 0
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg,#f093fb,#f5576c)">
            <h2>{count}</h2>
            <p>✅ Present Today</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        trained = "✅ Ready" if os.path.isfile(TRAINER_YML) else "❌ Not Trained"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg,#4facfe,#00f2fe)">
            <h2 style="font-size:1.5rem">{trained}</h2>
            <p>🤖 Model Status</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Registered students table
    st.subheader("👥 Registered Students")
    df = load_students()
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No students registered yet. Go to **Register New Student** to add.")

    # Today's attendance
    st.subheader("📋 Today's Attendance")
    today_df = load_attendance_today()
    if not today_df.empty:
        st.dataframe(today_df, use_container_width=True, hide_index=True)
    else:
        st.info("No attendance marked today. Go to **Take Attendance** to begin.")

# ════════════════════════════════════════════════════════════════
# PAGE 2: REGISTER NEW STUDENT
# ════════════════════════════════════════════════════════════════
elif page == "📸 Register New Student":
    st.header("📸 Register New Student")

    col1, col2 = st.columns([1, 1])
    with col1:
        student_id   = st.text_input("🆔 Student ID", placeholder="e.g. 201")
        student_name = st.text_input("👤 Student Name", placeholder="e.g. Rahul Sharma")

    with col2:
        st.markdown("#### 📷 Capture Face Photos")
        st.info("📌 Take **5 clear photos** from different angles.\nMake sure your face is well-lit and clearly visible.")
        num_photos = st.slider("Number of photos to capture", 3, 10, 5)

    if student_id and student_name:
        # Clean name for file system
        clean_name = "".join(c for c in student_name if c.isalnum() or c in (" ", "_", "-")).strip()

        st.markdown("---")
        st.subheader("📷 Take Photos One by One")
        st.warning("⚠️ Allow camera access in your browser when prompted.")

        # Store captured photos count in session
        if "captured_count" not in st.session_state:
            st.session_state.captured_count = 0
        if "last_registered" not in st.session_state:
            st.session_state.last_registered = None

        col_cam, col_status = st.columns([2, 1])

        with col_cam:
            img_file = st.camera_input(
                f"📸 Photo {st.session_state.captured_count + 1} of {num_photos}"
            )

        with col_status:
            if st.session_state.captured_count > 0:
                st.markdown(f"""
                <div class="success-box">
                    ✅ {st.session_state.captured_count} photo(s) captured!<br>
                    Need {max(0, num_photos - st.session_state.captured_count)} more.
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Click photo to begin registration")

            if st.session_state.captured_count >= num_photos:
                st.success(f"🎉 All {num_photos} photos done!\nClick **Save Profile** below.")

        if img_file is not None:
            # Process captured image
            img_arr = np.frombuffer(img_file.getvalue(), np.uint8)
            frame   = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)

            det     = detector()
            faces   = det.detectMultiScale(frame, 1.3, 5)

            if len(faces) == 0:
                st.error("❌ No face detected! Please try again with better lighting or closer to camera.")
            else:
                # Save each detected face
                serial = next_serial()
                for (x, y, w, h) in faces[:1]:  # only first face
                    sample_num = st.session_state.captured_count + 1
                    fname = f"{clean_name}.{serial}.{student_id}.{sample_num}.jpg"
                    fpath = os.path.join(TRAINING_DIR, " " + fname)
                    cv2.imwrite(fpath, frame[y:y+h, x:x+w])

                st.session_state.captured_count += 1

                # Show detected face
                frame_rgb = cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB
                )
                for (x, y, w, h) in faces[:1]:
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame_rgb, f"Face #{sample_num}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                st.image(frame_rgb, caption=f"✅ Face detected & saved (Photo {sample_num})", use_container_width=True)

        # Save Profile Button
        if st.session_state.captured_count >= num_photos:
            st.markdown("---")
            col_save, col_reset = st.columns(2)

            with col_save:
                if st.button("💾 Save Profile & Train Model", type="primary", use_container_width=True):
                    serial = next_serial()
                    save_student(serial, student_id, student_name)
                    with st.spinner("🤖 Training face recognition model..."):
                        success, count = train_model()
                    if success:
                        st.balloons()
                        st.markdown(f"""
                        <div class="success-box">
                            🎉 Profile saved successfully!<br>
                            👤 {student_name} (ID: {student_id}) registered.<br>
                            🤖 Model trained with {count} student(s).
                        </div>""", unsafe_allow_html=True)
                        st.session_state.captured_count = 0
                        st.session_state.last_registered = student_name
                    else:
                        st.error("❌ Training failed. No face images found.")

            with col_reset:
                if st.button("🔄 Reset / Register Another", use_container_width=True):
                    st.session_state.captured_count = 0
                    st.rerun()
    else:
        st.info("👆 Please fill in Student ID and Student Name to begin.")

# ════════════════════════════════════════════════════════════════
# PAGE 3: TAKE ATTENDANCE
# ════════════════════════════════════════════════════════════════
elif page == "✅ Take Attendance":
    st.header("✅ Take Attendance")

    if not os.path.isfile(TRAINER_YML):
        st.error("❌ Model not trained yet! Please register students and save their profile first.")
        st.stop()

    if not os.path.isfile(STUDENT_CSV):
        st.error("❌ No student data found. Please register students first.")
        st.stop()

    st.info("📌 Look at the camera clearly. Your face will be recognized automatically.")

    col_cam, col_result = st.columns([2, 1])

    with col_cam:
        cam_img = st.camera_input("📷 Look at camera to mark attendance")

    with col_result:
        st.subheader("📋 Recognition Result")

        if cam_img is not None:
            img_arr  = np.frombuffer(cam_img.getvalue(), np.uint8)
            frame    = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces_rect, results = recognize_face(gray)

            if results is None:
                st.error("Model not found.")
            elif len(results) == 0:
                st.warning("❌ No face detected. Try again.")
            else:
                marked = []
                for item in results:
                    x, y, w, h, name, sid, conf = item
                    if name != "Unknown":
                        color = (0, 255, 0)
                        mark_attendance(sid, name)
                        marked.append((sid, name))
                        st.markdown(f"""
                        <div class="success-box">
                            ✅ <b>{name}</b><br>
                            🆔 ID: {sid}<br>
                            📅 {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}<br>
                            🎯 Confidence: {100 - int(conf):.0f}%
                        </div>""", unsafe_allow_html=True)
                    else:
                        color = (255, 0, 0)
                        st.error("❌ Unknown person detected.")
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(frame_rgb, name, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                st.image(frame_rgb, caption="Recognized frame", use_container_width=True)

    # Today's attendance table
    st.markdown("---")
    st.subheader(f"📋 Today's Attendance — {datetime.datetime.now().strftime('%d %B %Y')}")
    today_df = load_attendance_today()
    if not today_df.empty:
        st.dataframe(today_df, use_container_width=True, hide_index=True)
        # Download button
        csv_data = today_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Today's Attendance CSV",
            csv_data,
            f"Attendance_{datetime.datetime.now().strftime('%d-%m-%Y')}.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        st.info("No attendance marked today yet.")

# ════════════════════════════════════════════════════════════════
# PAGE 4: ATTENDANCE RECORDS
# ════════════════════════════════════════════════════════════════
elif page == "📊 Attendance Records":
    st.header("📊 Attendance Records")

    files = sorted(
        [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")],
        reverse=True
    ) if os.path.isdir(ATTENDANCE_DIR) else []

    if not files:
        st.info("No attendance records found yet.")
    else:
        selected = st.selectbox("📅 Select Date", files)
        if selected:
            df = pd.read_csv(os.path.join(ATTENDANCE_DIR, selected))
            st.markdown(f"**{len(df)} record(s) found**")
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                csv_data,
                selected,
                "text/csv",
                use_container_width=True,
            )

# ───────────────────────── FOOTER ──────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#666;font-size:0.85rem;'>🎓 Face Recognition Attendance System | Built with OpenCV + Streamlit</center>",
    unsafe_allow_html=True
)


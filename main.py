############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2, os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

############################################# FUNCTIONS ################################################

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

##################################################################################

def tick():
    time_string = time.strftime('%I:%M:%S %p')
    date_string = time.strftime('%A, %d %B %Y')
    clock.config(text=f"{date_string}   |   {time_string}")
    clock.after(1000, tick)

###################################################################################

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(title='File Missing', message='haarcascade_frontalface_default.xml not found! Please check.')
        window.destroy()

###################################################################################

def save_pass():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        with open("TrainingImageLabel\psd.txt", "r") as tf:
            key = tf.read().strip()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
            return
        else:
            with open("TrainingImageLabel\psd.txt", "w") as tf:
                tf.write(new_pas.strip())
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    op = old.get().strip()
    newp = new.get().strip()
    nnewp = nnew.get().strip()
    if (op == key):
        if(newp == nnewp):
            with open("TrainingImageLabel\psd.txt", "w") as txf:
                txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()

###################################################################################

def change_pass():
    global master, old, new, nnew
    master = tk.Toplevel(window)
    master.geometry("440x220")
    master.resizable(False, False)
    master.title("Change Admin Password")
    master.configure(background="#f8fafc")

    lbl_title = tk.Label(master, text="Update Security Password", bg="#f8fafc", fg="#0f172a", font=('Segoe UI', 12, 'bold'))
    lbl_title.pack(pady=(12, 10))

    frame_inputs = tk.Frame(master, bg="#f8fafc")
    frame_inputs.pack(padx=20, pady=5)

    tk.Label(frame_inputs, text='Old Password:', bg='#f8fafc', fg='#334155', font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=6)
    old = tk.Entry(frame_inputs, width=22, fg="#0f172a", bg="#ffffff", relief='solid', bd=1, font=('Segoe UI', 10), show='*')
    old.grid(row=0, column=1, padx=10, pady=6)

    tk.Label(frame_inputs, text='New Password:', bg='#f8fafc', fg='#334155', font=('Segoe UI', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=6)
    new = tk.Entry(frame_inputs, width=22, fg="#0f172a", bg="#ffffff", relief='solid', bd=1, font=('Segoe UI', 10), show='*')
    new.grid(row=1, column=1, padx=10, pady=6)

    tk.Label(frame_inputs, text='Confirm Password:', bg='#f8fafc', fg='#334155', font=('Segoe UI', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=6)
    nnew = tk.Entry(frame_inputs, width=22, fg="#0f172a", bg="#ffffff", relief='solid', bd=1, font=('Segoe UI', 10), show='*')
    nnew.grid(row=2, column=1, padx=10, pady=6)

    frame_btns = tk.Frame(master, bg="#f8fafc")
    frame_btns.pack(pady=(14, 10))

    save1 = tk.Button(frame_btns, text="Save Password", command=save_pass, fg="white", bg="#2563eb", activebackground="#1d4ed8", activeforeground="white", relief='flat', font=('Segoe UI', 10, 'bold'), padx=15, pady=4, cursor='hand2')
    save1.pack(side='left', padx=10)

    cancel = tk.Button(frame_btns, text="Cancel", command=master.destroy, fg="#334155", bg="#e2e8f0", activebackground="#cbd5e1", relief='flat', font=('Segoe UI', 10, 'bold'), padx=15, pady=4, cursor='hand2')
    cancel.pack(side='left', padx=10)

#####################################################################################

def psw():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        with open("TrainingImageLabel\psd.txt", "r") as tf:
            key = tf.read().strip()
    else:
        new_pas = tsd.askstring('Setup Password', 'Please set an Admin Password:', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again.')
            return
        else:
            with open("TrainingImageLabel\psd.txt", "w") as tf:
                tf.write(new_pas.strip())
            mess._show(title='Password Saved', message='Admin password was set successfully!!')
            return
    password = tsd.askstring('Admin Verification', 'Enter Admin Password to save profile:', show='*')
    if (password == key):
        TrainImages()
    elif (password == None):
        pass
    else:
        mess._show(title='Access Denied', message='Wrong password entered! Please try again.')

######################################################################################

def clear():
    txt.delete(0, 'end')
    message1.configure(text="Step 1: Capture Face Samples   >>>   Step 2: Save Profile", fg="#1e40af", bg="#eff6ff")

def clear2():
    txt2.delete(0, 'end')
    message1.configure(text="Step 1: Capture Face Samples   >>>   Step 2: Save Profile", fg="#1e40af", bg="#eff6ff")

#######################################################################################

def TakeImages():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
    else:
        with open("StudentDetails\StudentDetails.csv", 'a+', newline='') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1

    Id = txt.get().strip()
    name = txt2.get().strip()

    if not Id:
        message.configure(text="Please Enter Student ID", fg="#b91c1c", bg="#fef2f2")
        return
    if not name or not (name.replace(" ", "").isalpha()):
        message.configure(text="Enter a Valid Alphabetical Name", fg="#b91c1c", bg="#fef2f2")
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess._show(title='Camera Error', message='Could not open webcam. Please verify your camera device.')
        return

    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    message1.configure(text="Capturing face samples... Look at webcam (Press 'q' to stop)", fg="#15803d", bg="#f0fdf4")
    window.update()

    while True:
        ret, img = cam.read()
        if not ret or img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (37, 99, 235), 2)
            sampleNum = sampleNum + 1
            cv2.imwrite("TrainingImage\ " + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                        gray[y:y + h, x:x + w])
            cv2.putText(img, f"Sample: {sampleNum}/100", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (37, 99, 235), 2)
            cv2.imshow('Face Enrollment - Press Q to Cancel', img)

        if cv2.waitKey(80) & 0xFF == ord('q'):
            break
        elif sampleNum >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    if sampleNum > 0:
        row = [serial, '', Id, '', name]
        with open('StudentDetails\StudentDetails.csv', 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        res = f"100 Samples Captured for ID: {Id} ({name})"
        message1.configure(text=res, fg="#15803d", bg="#f0fdf4")
        message.configure(text="Click 'Save Profile' below to train biometric model", fg="#1e40af", bg="#eff6ff")
    else:
        message1.configure(text="No samples captured. Please try again.", fg="#b91c1c", bg="#fef2f2")

########################################################################################

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ID = getImagesAndLabels("TrainingImage")
    if len(faces) == 0:
        mess._show(title='No Biometric Data', message='Please register and capture face images first!')
        return
    try:
        recognizer.train(faces, np.array(ID))
        recognizer.save("TrainingImageLabel\Trainner.yml")
    except Exception as e:
        mess._show(title='Training Error', message=f'Failed to train model: {e}')
        return

    count = len(set(ID))
    message1.configure(text="Biometric Profile Trained Successfully!", fg="#15803d", bg="#f0fdf4")
    message.configure(text=f"Total Registered Profiles: {count}", fg="#15803d", bg="#f0fdf4")
    mess._show(title='Model Ready', message='Biometric Profile trained and saved successfully!')

############################################################################################

def getImagesAndLabels(path):
    if not os.path.exists(path):
        return [], []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".jpg")]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            ID = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(ID)
        except Exception:
            pass
    return faces, Ids

###########################################################################################

def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")

    for k in tv.get_children():
        tv.delete(k)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    exists3 = os.path.isfile("TrainingImageLabel\Trainner.yml")
    if exists3:
        recognizer.read("TrainingImageLabel\Trainner.yml")
    else:
        mess._show(title='Model Not Found', message='Please register students and click "Save Profile" first!')
        return

    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess._show(title='Camera Error', message='Could not open webcam. Please verify your camera device.')
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    exists1 = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails\StudentDetails.csv")
    else:
        mess._show(title='Details Missing', message='StudentDetails.csv missing, please register students!')
        cam.release()
        cv2.destroyAllWindows()
        return

    attendance = []
    recognized_names = set()

    while True:
        ret, im = cam.read()
        if not ret or im is None:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 60):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID_str = str(ID[0]) if len(ID) > 0 else "Unknown"
                Name_str = str(aa[0]) if len(aa) > 0 else "Unknown"

                attendance = [ID_str, '', Name_str, '', str(date), '', str(timeStamp)]
                recognized_names.add(Name_str)

                # Visual indicators
                cv2.rectangle(im, (x, y), (x + w, y + h), (16, 185, 129), 2)
                cv2.putText(im, f"{Name_str} ({int(100 - conf)}%)", (x, y - 10), font, 0.7, (16, 185, 129), 2)
            else:
                cv2.rectangle(im, (x, y), (x + w, y + h), (239, 68, 68), 2)
                cv2.putText(im, "Unknown", (x, y - 10), font, 0.7, (239, 68, 68), 2)

        cv2.putText(im, "Press 'Q' to Finish Attendance", (20, 30), font, 0.7, (255, 255, 255), 2)
        cv2.imshow('Face Attendance Scanner - Press Q to Complete', im)
        if cv2.waitKey(1) == ord('q'):
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    if attendance:
        file_path = "Attendance\Attendance_" + date + ".csv"
        exists = os.path.isfile(file_path)
        with open(file_path, 'a+', newline='') as csvFile1:
            writer = csv.writer(csvFile1)
            if not exists:
                writer.writerow(col_names)
            writer.writerow(attendance)

    # Reload Table
    file_path = "Attendance\Attendance_" + date + ".csv"
    if os.path.isfile(file_path):
        with open(file_path, 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            i = 0
            for lines in reader1:
                i = i + 1
                if i > 1 and len(lines) >= 7:
                    if lines[0].strip() and lines[0] != 'Id':
                        tv.insert('', 0, text=lines[0], values=(lines[2], lines[4], lines[6]))

    cam.release()
    cv2.destroyAllWindows()
    if recognized_names:
        mess._show(title='Attendance Logged', message=f'Successfully marked attendance for:\n{", ".join(recognized_names)}')

######################################## GUI FRONT-END ###########################################

window = tk.Tk()
window.geometry("1280x740")
window.resizable(False, False)
window.title("FaceAttend Pro - Biometric Attendance Monitoring System")
window.configure(background='#f8fafc')
window.attributes('-topmost', True)
window.update()
window.attributes('-topmost', False)

# ─────────────────── MODERN TOP NAVBAR ───────────────────
header = tk.Frame(window, bg="#0f172a", height=85)
header.pack(fill='x', side='top')
header.pack_propagate(False)

header_content = tk.Frame(header, bg="#0f172a")
header_content.pack(fill='both', expand=True, padx=30)

brand_frame = tk.Frame(header_content, bg="#0f172a")
brand_frame.pack(side='left', pady=14)

title_lbl = tk.Label(brand_frame, text="FaceAttend Pro", fg="#ffffff", bg="#0f172a", font=('Segoe UI', 18, 'bold'))
title_lbl.pack(anchor='w')

subtitle_lbl = tk.Label(brand_frame, text="Biometric Attendance & Automated Optical Identity System", fg="#94a3b8", bg="#0f172a", font=('Segoe UI', 9))
subtitle_lbl.pack(anchor='w')

# Clock badge on right
clock_box = tk.Frame(header_content, bg="#1e293b", padx=16, pady=8, highlightbackground="#334155", highlightthickness=1)
clock_box.pack(side='right', pady=18)

clock = tk.Label(clock_box, fg="#38bdf8", bg="#1e293b", font=('Segoe UI', 10, 'bold'))
clock.pack()
tick()

# ─────────────────── MAIN CONTAINER ───────────────────
body = tk.Frame(window, bg="#f8fafc")
body.pack(fill='both', expand=True, padx=30, pady=20)

# Configure Grid for 2 Cards (Equal width, 20px gap)
body.columnconfigure(0, weight=1, uniform="card")
body.columnconfigure(1, weight=1, uniform="card")
body.rowconfigure(0, weight=1)

# ═══════════════════ LEFT CARD: ATTENDANCE ═══════════════════
card_left = tk.Frame(body, bg="#ffffff", highlightbackground="#e2e8f0", highlightthickness=1, bd=0)
card_left.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

# Card Header
card1_hdr = tk.Frame(card_left, bg="#2563eb", height=42)
card1_hdr.pack(fill='x')
card1_hdr.pack_propagate(False)
tk.Label(card1_hdr, text="  ATTENDANCE VERIFICATION & LOGS", fg="#ffffff", bg="#2563eb", font=('Segoe UI', 11, 'bold'), anchor='w').pack(fill='both', expand=True, padx=15)

card1_body = tk.Frame(card_left, bg="#ffffff", padx=20, pady=20)
card1_body.pack(fill='both', expand=True)

trackImg = tk.Button(
    card1_body,
    text="Launch Camera & Mark Attendance",
    command=TrackImages,
    fg="white",
    bg="#10b981",
    activebackground="#059669",
    activeforeground="white",
    relief='flat',
    bd=0,
    font=('Segoe UI', 12, 'bold'),
    height=2,
    cursor='hand2'
)
trackImg.pack(fill='x', pady=(0, 18))

tk.Label(card1_body, text="Today's Attendance Stream", bg="#ffffff", fg="#0f172a", font=('Segoe UI', 11, 'bold'), anchor='w').pack(fill='x', pady=(0, 8))

# Treeview Style
style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview",
    background="#ffffff",
    foreground="#0f172a",
    rowheight=28,
    fieldbackground="#ffffff",
    font=('Segoe UI', 10)
)
style.configure("Treeview.Heading",
    background="#f1f5f9",
    foreground="#334155",
    font=('Segoe UI', 10, 'bold'),
    borderwidth=1,
    relief='flat'
)
style.map("Treeview", background=[('selected', '#e0f2fe')], foreground=[('selected', '#0369a1')])

# Table Frame with Scrollbar
table_frame = tk.Frame(card1_body, bg="#ffffff")
table_frame.pack(fill='both', expand=True)

tv = ttk.Treeview(table_frame, columns=('name', 'date', 'time'), show='tree headings', height=11)
tv.column('#0', width=90, anchor='center')
tv.column('name', width=160, anchor='w')
tv.column('date', width=110, anchor='center')
tv.column('time', width=110, anchor='center')

tv.heading('#0', text='STUDENT ID')
tv.heading('name', text='STUDENT NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

scroll = ttk.Scrollbar(table_frame, orient='vertical', command=tv.yview)
tv.configure(yscrollcommand=scroll.set)
scroll.pack(side='right', fill='y')
tv.pack(side='left', fill='both', expand=True)

# Load existing today's attendance into table
today_date = datetime.datetime.now().strftime('%d-%m-%Y')
file_today = "Attendance\Attendance_" + today_date + ".csv"
if os.path.isfile(file_today):
    try:
        with open(file_today, 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            i = 0
            for lines in reader1:
                i = i + 1
                if i > 1 and len(lines) >= 7:
                    if lines[0].strip() and lines[0] != 'Id':
                        tv.insert('', 0, text=lines[0], values=(lines[2], lines[4], lines[6]))
    except Exception:
        pass

quitWindow = tk.Button(
    card1_body,
    text="Exit Application",
    command=window.destroy,
    fg="#64748b",
    bg="#f8fafc",
    activebackground="#fee2e2",
    activeforeground="#b91c1c",
    relief='solid',
    bd=1,
    font=('Segoe UI', 10, 'bold'),
    cursor='hand2',
    pady=6
)
quitWindow.pack(fill='x', pady=(15, 0))

# ═══════════════════ RIGHT CARD: REGISTRATION ═══════════════════
card_right = tk.Frame(body, bg="#ffffff", highlightbackground="#e2e8f0", highlightthickness=1, bd=0)
card_right.grid(row=0, column=1, sticky="nsew", padx=(15, 0))

# Card Header
card2_hdr = tk.Frame(card_right, bg="#4f46e5", height=42)
card2_hdr.pack(fill='x')
card2_hdr.pack_propagate(False)
tk.Label(card2_hdr, text="  NEW STUDENT BIOMETRIC ENROLLMENT", fg="#ffffff", bg="#4f46e5", font=('Segoe UI', 11, 'bold'), anchor='w').pack(fill='both', expand=True, padx=15)

card2_body = tk.Frame(card_right, bg="#ffffff", padx=25, pady=20)
card2_body.pack(fill='both', expand=True)

# Input 1: Student ID
tk.Label(card2_body, text="Student ID / Roll Number", bg="#ffffff", fg="#334155", font=('Segoe UI', 10, 'bold'), anchor='w').pack(fill='x', pady=(0, 4))
row1 = tk.Frame(card2_body, bg="#ffffff")
row1.pack(fill='x', pady=(0, 15))
txt = tk.Entry(row1, fg="#0f172a", bg="#f8fafc", relief='solid', bd=1, font=('Segoe UI', 11))
txt.pack(side='left', fill='x', expand=True, ipady=5)
clearButton = tk.Button(row1, text="Clear", command=clear, fg="#475569", bg="#f1f5f9", activebackground="#e2e8f0", relief='solid', bd=1, font=('Segoe UI', 9, 'bold'), padx=14, cursor='hand2')
clearButton.pack(side='right', padx=(10, 0), ipady=3)

# Input 2: Student Name
tk.Label(card2_body, text="Full Legal Name", bg="#ffffff", fg="#334155", font=('Segoe UI', 10, 'bold'), anchor='w').pack(fill='x', pady=(0, 4))
row2 = tk.Frame(card2_body, bg="#ffffff")
row2.pack(fill='x', pady=(0, 15))
txt2 = tk.Entry(row2, fg="#0f172a", bg="#f8fafc", relief='solid', bd=1, font=('Segoe UI', 11))
txt2.pack(side='left', fill='x', expand=True, ipady=5)
clearButton2 = tk.Button(row2, text="Clear", command=clear2, fg="#475569", bg="#f1f5f9", activebackground="#e2e8f0", relief='solid', bd=1, font=('Segoe UI', 9, 'bold'), padx=14, cursor='hand2')
clearButton2.pack(side='right', padx=(10, 0), ipady=3)

# Step Instructions
message1 = tk.Label(
    card2_body,
    text="Step 1: Capture Face Samples   >>>   Step 2: Save Profile",
    bg="#eff6ff",
    fg="#1e40af",
    font=('Segoe UI', 10, 'bold'),
    bd=1,
    relief='solid',
    pady=10
)
message1.pack(fill='x', pady=(5, 18))

# Action Button 1: Capture Images
takeImg = tk.Button(
    card2_body,
    text="1. Capture Face Samples (100 Frames)",
    command=TakeImages,
    fg="white",
    bg="#4f46e5",
    activebackground="#4338ca",
    activeforeground="white",
    relief='flat',
    bd=0,
    font=('Segoe UI', 11, 'bold'),
    height=2,
    cursor='hand2'
)
takeImg.pack(fill='x', pady=(0, 12))

# Action Button 2: Save Profile & Train
trainImg = tk.Button(
    card2_body,
    text="2. Save Profile & Train Model",
    command=psw,
    fg="white",
    bg="#0284c7",
    activebackground="#0369a1",
    activeforeground="white",
    relief='flat',
    bd=0,
    font=('Segoe UI', 11, 'bold'),
    height=2,
    cursor='hand2'
)
trainImg.pack(fill='x', pady=(0, 18))

# Total Count Badge
res = 0
exists = os.path.isfile("StudentDetails\StudentDetails.csv")
if exists:
    try:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                res = res + 1
        res = max(0, (res // 2) - 1)
    except Exception:
        res = 0

message = tk.Label(
    card2_body,
    text=f"Total Registered Students: {res}",
    bg="#f0fdf4",
    fg="#15803d",
    font=('Segoe UI', 11, 'bold'),
    bd=1,
    relief='solid',
    pady=10
)
message.pack(fill='x')

# ─────────────────── MENUBAR ───────────────────
menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Admin Password', command=change_pass)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='Settings', menu=filemenu)

window.configure(menu=menubar)

####################################################################################################
if __name__ == "__main__":
    window.mainloop()

# AttendX — Institutional Attendance Management System

AI-powered multi-role attendance platform with face recognition, liveness detection,
and comprehensive analytics for administrators, faculty, and students.

---

## Quick Start

```bash
# 1. Install dependencies (minimal set to get running)
pip install flask loguru opencv-python numpy

# 2. Run (auto-seeds demo data)
python main_web.py

# 3. Open browser
http://localhost:5000
```

### Demo Accounts
| Role    | Email                    | Password    |
|---------|--------------------------|-------------|
| Admin   | admin@school.edu         | admin123    |
| Teacher | teacher@school.edu       | teacher123  |
| Student | student@school.edu       | student123  |

---

## Feature Summary

### Admin Dashboard
- **Students table** — roll no, name, branch, semester, section, photo, face enrollment status
- **Faculty table** — employee ID, name, branch, designation, qualifications
- **Courses table** — code, name, branch, semester, type, credits, assigned faculty, enrolled count
- **Attendance sheets** — filterable by date range, branch, semester; shows all fields
- **Analytics** — branch-wise attendance %, at-risk student list, 14-day trend
- **Photo upload** → auto face embedding generation (when InsightFace is installed)
- **Audit log** — every add/edit/delete is tracked

### Teacher Dashboard
- **Live camera** — MJPEG stream with bounding boxes and recognition chips
- **Live register** — real-time attendance register updates as students are recognised
- **Class roster** — full roster view with manual override capability
- **Reports** — course-wise attendance summary and export

### Student Dashboard
- **Today status hero** — clear Present/Absent indicator with date
- **Subject-wise bars** — progress bars with 75% threshold line
- **Subject detail cards** — attended/missed/total + "can miss N more" calculation
- **Full history table** — filterable by month and subject

---

## Full Installation (with Face Recognition)

```bash
pip install -r requirements.txt

# Download InsightFace models (~300 MB, one-time)
python scripts/setup.py

# Enroll a new face (optional — admin can upload photos through UI)
python scripts/enroll.py --name "Jane Doe" --role student

# Run
python main_web.py
```

---

## Architecture

```
main_web.py          ← Entry point: DB init, CV pipeline, Flask server
│
├── database/
│   ├── schema.py    ← Full relational schema (students, faculty, courses, attendance…)
│   └── db.py        ← Thread-local SQLite manager with WAL mode
│
├── auth/
│   └── manager.py   ← PBKDF2 password hashing, session management, RBAC
│
├── web_app/
│   ├── server.py    ← All REST API routes (admin / teacher / student)
│   ├── templates/
│   │   ├── login.html
│   │   ├── admin.html
│   │   ├── teacher.html
│   │   └── student.html
│   └── static/css/app.css   ← DM Sans design system
│
├── core/
│   ├── camera.py    ← Threaded camera capture
│   └── pipeline.py  ← Inference pipeline (detect → track → embed → search → log)
│
├── vision/
│   ├── detector.py  ← RetinaFace (InsightFace) + OpenCV fallback
│   ├── tracker.py   ← DeepSORT-style multi-face tracker
│   └── aligner.py   ← 5-point affine alignment to ArcFace 112×112 template
│
├── recognition/
│   ├── embedder.py  ← ArcFace embedding extraction
│   └── searcher.py  ← FAISS (or NumPy brute-force) vector search
│
├── security/
│   └── antispoofing.py ← LBP + Fourier + blink liveness detection
│
└── analytics/
    └── reports.py   ← CSV/Excel export, attendance % calculations
```

---

## Database Schema

```
students    → roll_no, name, branch, semester, section, photo, embedding, user_id
faculty     → employee_id, name, branch, designation, photo, embedding, user_id
courses     → code, name, branch, semester, credits, type, faculty_id
enrollments → student_id ↔ course_id (many-to-many)
attendance  → student_id, course_id, date, entry/exit time, confidence, liveness, status
users       → email, password_hash, role (links to students/faculty)
sessions    → token-based auth
audit_log   → every admin action
devices     → registered cameras
```

---

## Environment Variables

```bash
CAMERA_DEVICE_ID=0          # Camera index (default: 0)
RECOGNITION_THRESHOLD=0.45  # Cosine similarity threshold
COOLDOWN_MINUTES=5          # Min gap between duplicate attendance entries
DISABLE_ANTISPOOFING=1      # Bypass liveness check for testing
```
# Authface
# Authface

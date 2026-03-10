# AttendX — AI-Powered Attendance Management System

> **Local software** — download, install, and run on your own machine with a webcam.

AttendX is a full-featured multi-role attendance platform with **face recognition**, **liveness detection**, and rich analytics for administrators, teachers, and students.

---

## ⚡ Quick Start (3 Steps)

### 1 — Download

```bash
git clone https://github.com/soumyasswt/AttendX.git
cd AttendX
```

Or click **Code → Download ZIP** on GitHub.

---

### 2 — Setup

**Windows:**
```
Double-click setup.bat
```

**Linux / macOS:**
```bash
chmod +x setup.sh run.sh
./setup.sh
```

The setup script will:
- Create a Python virtual environment (`venv/`)
- Install all required packages
- Optionally install face recognition (InsightFace, ~400 MB)
- Create the `data/`, `uploads/`, `logs/` directories

---

### 3 — Run

**Windows:**
```
Double-click run.bat
```

**Linux / macOS:**
```bash
./run.sh
```

**Or manually:**
```bash
python main_web.py
```

Then open **http://localhost:5000** in your browser.

---

## 🔑 Demo Accounts.

| Role    | Email                    | Password  |
|---------|--------------------------|-----------|
| Admin   | admin@school.edu         | admin     |
| Teacher | teacher@school.edu       | teacher   |
| Student | student@school.edu       | student   |

> Demo data is **auto-seeded** on first run — students, faculty, courses, and sample attendance records.

---

## 📋 System Requirements

| Requirement | Details |
|-------------|---------|
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| **Python** | 3.10, 3.11, or 3.12 |
| **Camera** | Any USB or built-in webcam (for live recognition) |
| **RAM** | 4 GB minimum, 8 GB recommended (for face recognition models) |
| **Disk** | ~600 MB (including InsightFace models) |

---

## 🧠 Face Recognition Setup (Optional)

The app works **without** face recognition (manual attendance marking is available). To enable AI face recognition:

1. Run `setup.bat` / `setup.sh` and answer **Y** when asked about face recognition packages.
2. That automatically downloads the InsightFace `buffalo_l` model pack (~300 MB).
3. Enroll student faces via the **Admin → Students → Upload Photo** button in the UI.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

Key settings:

```env
CAMERA_DEVICE_ID=0        # 0 = default webcam, 1 = second camera
RECOGNITION_THRESHOLD=0.45 # Lower = more permissive
COOLDOWN_MINUTES=5         # Prevent duplicate scans
DISABLE_ANTISPOOFING=0     # Set to 1 for testing
PORT=5000
```

---

## ✨ Features

### Admin Dashboard
- Manage students, faculty, courses
- View attendance sheets with filters (date, branch, semester)
- Analytics: branch-wise stats, 14-day trend, at-risk students
- Photo upload → auto face embedding enrollment
- Full audit log

### Teacher Dashboard
- **Live camera feed** with face recognition overlays
- Real-time attendance register (auto-updates as students are recognised)
- Manual mark override
- Class roster with attendance history
- Course-wise reports and export

### Student Dashboard
- Today's attendance status (Present/Absent)
- Subject-wise attendance % with 75% threshold indicator
- "Can miss N more classes" calculator
- Full attendance history with filters

---

## 🗂️ Project Structure

```
attendx/
├── main_web.py          ← Entry point
├── setup.bat / setup.sh ← One-click setup
├── run.bat / run.sh     ← One-click run
│
├── database/            ← SQLite schema & connection manager
├── auth/                ← Login, sessions, RBAC
├── core/                ← Camera capture & inference pipeline
├── vision/              ← Face detection, tracking, alignment
├── recognition/         ← ArcFace embeddings & FAISS search
├── security/            ← Liveness / anti-spoofing
├── analytics/           ← Reports & Excel export
│
└── web_app/
    ├── server.py        ← All REST API routes
    └── templates/       ← HTML dashboards (login, admin, teacher, student)
```

---

## 🛠️ Manual Installation (without setup script)

```bash
# 1. Clone
git clone https://github.com/soumyasswt/AttendX.git
cd AttendX

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install flask loguru opencv-python numpy gunicorn openpyxl pandas

# 4. (Optional) Face recognition
pip install insightface onnxruntime
python scripts/setup.py    # Downloads InsightFace models

# 5. Run
python main_web.py
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera shows "initialising…" | Wait ~20 s for models to load. If it stays, check no other app is using the camera. |
| Camera shows "not available" | Try setting `CAMERA_DEVICE_ID=1` in `.env` |
| `ModuleNotFoundError` | Run `setup.bat` again, or `pip install -r requirements.txt` in your venv |
| Port 5000 already in use | Set `PORT=5001` in `.env` |
| Face not recognised | Upload a clear, well-lit front-facing photo via Admin → Upload Photo |

---

## 📄 License

MIT — free to use, modify, and distribute.

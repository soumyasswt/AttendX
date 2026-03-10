"""
main_web.py
===========
AttendX Platform — Full Multi-Role Institutional Attendance System

Start with:
    python main_web.py

Opens at: http://localhost:5000

Demo accounts (auto-seeded):
    admin@school.edu    / admin     → Admin Dashboard
    teacher@school.edu  / teacher   → Teacher Dashboard
    student@school.edu  / student   → Student Dashboard
"""

from __future__ import annotations
import queue, sys, threading
from pathlib import Path

from loguru import logger

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from database.db import DB
from database.schema import SCHEMA, BRANCHES_SEED, DEFAULT_DEVICE
from auth.manager import Auth


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def _setup_log():
    (ROOT/'logs').mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level='INFO', colorize=True,
               format='<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}')
    logger.add(str(ROOT/'logs'/'app_{time:YYYY-MM-DD}.log'), level='DEBUG',
               rotation='20 MB', retention='7 days', enqueue=True)


# ─────────────────────────────────────────────────────────────────────────────
# Database init + demo seeding
# ─────────────────────────────────────────────────────────────────────────────
def _init_db() -> DB:
    db = DB(str(ROOT / 'data' / 'attendx.db'))
    with db.tx() as c:
        c.executescript(SCHEMA)
        c.executescript(BRANCHES_SEED)
        c.execute(DEFAULT_DEVICE)
    _seed_demo(db)
    return db


def _seed_demo(db: DB) -> None:
    auth = Auth(db)

    demo_users = [
        ('admin@school.edu',   'admin',   'admin'),
        ('teacher@school.edu', 'teacher', 'teacher'),
        ('teacher2@school.edu','teacher', 'teacher'),
        ('student@school.edu', 'student', 'student'),
        ('student2@school.edu','student', 'student'),
        ('student3@school.edu','student', 'student'),
    ]
    for email, pw, role in demo_users:
        if not db.one("SELECT id FROM users WHERE email=?", (email,)):
            try:
                auth.create_user(email, pw, role)
                logger.debug("Seeded user: {} ({})", email, role)
            except Exception as e:
                logger.warning("Seed user {} failed: {}", email, e)

    # Seed faculty records
    faculty_data = [
        ('teacher@school.edu',  'FAC001', 'Dr. Sarah Mitchell', 'CSE',  'Associate Professor', 'PhD Computer Science', 12),
        ('teacher2@school.edu', 'FAC002', 'Prof. Rajan Kumar',   'ECE',  'Professor',           'M.Tech Electronics',   18),
    ]
    for email, emp_id, name, branch, desig, qual, exp in faculty_data:
        if not db.one("SELECT id FROM faculty WHERE employee_id=?", (emp_id,)):
            u = db.one("SELECT id FROM users WHERE email=?", (email,))
            if u:
                with db.tx() as c:
                    c.execute(
                        "INSERT INTO faculty (employee_id,name,email,branch,designation,qualification,experience_yrs,user_id) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (emp_id, name, email, branch, desig, qual, exp, u['id'])
                    )
                logger.debug("Seeded faculty: {}", name)

    # Seed students
    student_data = [
        ('student@school.edu',  'CSE21001', 'Alex Johnson',   'CSE', 5, 'A'),
        ('student2@school.edu', 'CSE21002', 'Priya Sharma',   'CSE', 5, 'A'),
        ('student3@school.edu', 'ECE21001', 'Mohammed Rafiq', 'ECE', 5, 'B'),
    ]
    for email, roll, name, branch, sem, sec in student_data:
        if not db.one("SELECT id FROM students WHERE roll_no=?", (roll,)):
            u = db.one("SELECT id FROM users WHERE email=?", (email,))
            with db.tx() as c:
                cur = c.execute(
                    "INSERT INTO students (roll_no,name,email,branch,semester,section,user_id) VALUES (?,?,?,?,?,?,?)",
                    (roll, name, email, branch, sem, sec, u['id'] if u else None)
                )
                stu_id = cur.lastrowid
            logger.debug("Seeded student: {}", name)
            _auto_enroll(db, stu_id, branch, sem)

    # Seed courses
    courses_data = [
        ('CS501', 'Machine Learning',        'CSE', 5, 3, 'Theory',    'FAC001'),
        ('CS502', 'Computer Networks',        'CSE', 5, 4, 'Theory',    'FAC001'),
        ('CS503', 'ML Lab',                   'CSE', 5, 2, 'Lab',       'FAC001'),
        ('EC501', 'Digital Signal Processing','ECE', 5, 4, 'Theory',    'FAC002'),
        ('EC502', 'VLSI Design',              'ECE', 5, 3, 'Theory',    'FAC002'),
    ]
    for code, name, branch, sem, credits, ctype, fac_empid in courses_data:
        if not db.one("SELECT id FROM courses WHERE code=?", (code,)):
            fac = db.one("SELECT id FROM faculty WHERE employee_id=?", (fac_empid,))
            with db.tx() as c:
                cur = c.execute(
                    "INSERT INTO courses (code,name,branch,semester,credits,type,faculty_id,academic_year) VALUES (?,?,?,?,?,?,?,?)",
                    (code, name, branch, sem, credits, ctype, fac['id'] if fac else None, '2024-25')
                )
                crs_id = cur.lastrowid
            logger.debug("Seeded course: {}", code)
            _auto_enroll_course(db, crs_id, branch, sem)

    # Seed a few attendance records so dashboards aren't empty
    _seed_sample_attendance(db)


def _auto_enroll(db: DB, stu_id: int, branch: str, semester: int) -> None:
    courses = db.all("SELECT id FROM courses WHERE branch=? AND semester=? AND active=1", (branch, semester))
    with db.tx() as c:
        for co in courses:
            c.execute("INSERT OR IGNORE INTO enrollments (student_id,course_id) VALUES (?,?)", (stu_id, co['id']))


def _auto_enroll_course(db: DB, course_id: int, branch: str, semester: int) -> None:
    students = db.all("SELECT id FROM students WHERE branch=? AND semester=? AND active=1", (branch, semester))
    with db.tx() as c:
        for s in students:
            c.execute("INSERT OR IGNORE INTO enrollments (student_id,course_id) VALUES (?,?)", (s['id'], course_id))


def _seed_sample_attendance(db: DB) -> None:
    """Seed a week of sample attendance so dashboards show data."""
    from datetime import date, timedelta
    import random

    students = db.all("SELECT id FROM students WHERE active=1")
    courses  = db.all("SELECT id FROM courses WHERE active=1")
    if not students or not courses:
        return

    for day_offset in range(7, 0, -1):
        d = (date.today() - timedelta(days=day_offset)).isoformat()
        for stu in students:
            enrolments = db.all("SELECT course_id FROM enrollments WHERE student_id=?", (stu['id'],))
            for enr in enrolments:
                if random.random() > 0.25:  # 75% chance of attendance
                    existing = db.one(
                        "SELECT id FROM attendance WHERE student_id=? AND course_id=? AND date=? AND event_type='ENTRY'",
                        (stu['id'], enr['course_id'], d)
                    )
                    if not existing:
                        entry_h = random.randint(9,11)
                        entry_m = random.randint(0,59)
                        status  = 'LATE' if entry_h >= 10 else 'PRESENT'
                        entry_ts = f"{d}T{entry_h:02d}:{entry_m:02d}:00"
                        db.run(
                            "INSERT OR IGNORE INTO attendance (student_id,course_id,date,entry_time,status,event_type,recognition_confidence,liveness_score) "
                            "VALUES (?,?,?,?,'PRESENT','ENTRY',?,?)",
                            (stu['id'], enr['course_id'], d, entry_ts,
                             round(random.uniform(0.78, 0.99), 3),
                             round(random.uniform(0.82, 0.99), 3))
                        )


# ─────────────────────────────────────────────────────────────────────────────
# CV Pipeline (camera + face recognition)
# ─────────────────────────────────────────────────────────────────────────────
def _start_cv(db: DB, result_q: queue.Queue) -> None:
    """Load ML models and start camera + inference pipeline."""
    try:
        from config.settings import config
        from core.camera import CameraCapture
        from core.pipeline import InferencePipeline
        from vision.detector import FaceDetector
        from vision.tracker import FaceTracker
        from vision.aligner import FaceAligner
        from recognition.embedder import ArcFaceEmbedder
        from recognition.searcher import EmbeddingSearcher
        from security.antispoofing import LivenessDetector

        logger.info("Initialising face detection models…")
        detector  = FaceDetector(config)
        detector.initialize()
        embedder  = ArcFaceEmbedder(config)
        embedder.initialize(face_analysis=getattr(detector, 'face_analysis', None))
        searcher  = EmbeddingSearcher(config)

        # Load enrolled faces
        enrolled = db.all(
            "SELECT s.id, s.name, s.branch AS department, s.embedding "
            "FROM students s WHERE s.active=1 AND s.face_enrolled=1"
        )
        if enrolled:
            records = []
            for row in enrolled:
                if row['embedding']:
                    records.append({'user_id': row['id'], 'name': row['name'],
                                    'department': row['department'],
                                    'embedding': db.blob_to_embed(row['embedding'])})
            if records:
                searcher.add_faces_bulk(records)
                logger.success("Loaded {} face embeddings", len(records))
        else:
            logger.warning("No enrolled faces — recognition will return 'Unknown'")

        # Inject references into Flask server for photo upload enrollment
        from web_app.server import inject_ml
        inject_ml(detector, embedder, FaceAligner(config), searcher)

        camera   = CameraCapture(config)
        pipeline = InferencePipeline(
            cfg=config, camera=camera,
            detector=detector, tracker=FaceTracker(config),
            aligner=FaceAligner(config), embedder=embedder,
            searcher=searcher, liveness=LivenessDetector(config),
            db=db, result_queue=result_q,
        )
        camera.start()
        pipeline.start()
        logger.success("CV pipeline active ✓")

    except Exception as e:
        logger.error("CV pipeline failed to start: {}. Continuing without camera.", e)


# ─────────────────────────────────────────────────────────────────────────────
# Flask web server
# ─────────────────────────────────────────────────────────────────────────────
def _start_flask(db: DB, result_q: queue.Queue, host: str, port: int) -> None:
    from web_app.server import app, init_server
    uploads_dir = ROOT / 'uploads'
    init_server(db, result_q, uploads_dir)
    logger.success("AttendX → http://{}:{}", host, port)
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _setup_log()

    banner = """
    ╔══════════════════════════════════════════════════╗
    ║             AttendX  — Starting Up               ║
    ╠══════════════════════════════════════════════════╣
    ║  Admin    →  admin@school.edu    / admin123       ║
    ║  Teacher  →  teacher@school.edu  / teacher123     ║
    ║  Student  →  student@school.edu  / student123     ║
    ╚══════════════════════════════════════════════════╝
    """
    print(banner)

    # 1. Database
    logger.info("Initialising database…")
    db = _init_db()
    logger.success("Database ready: {}", ROOT/'data'/'attendx.db')

    # 2. Shared queue
    result_q: queue.Queue = queue.Queue(maxsize=10)

    # 3. CV pipeline (daemon thread — won't block exit)
    cv_thread = threading.Thread(target=_start_cv, args=(db, result_q), daemon=True, name='CVPipeline')
    cv_thread.start()

    # 4. Flask (blocks main thread)
    _start_flask(db, result_q, '0.0.0.0', 5000)


if __name__ == '__main__':
    main()

"""
database/schema.py
==================
Complete institutional schema:
  students  → roll_no, name, branch, semester, photo, embedding
  faculty   → employee_id, name, branch, subjects, photo, embedding
  courses   → code, name, branch, semester, faculty_id
  attendance→ student_id, course_id, date, times, confidence, liveness
  users     → auth table linking to student/faculty by role
  devices   → camera registry
  audit_log → every admin action tracked
"""

SCHEMA = """
PRAGMA foreign_keys = ON;

/* ── Branch / Semester reference data ─────────────────────── */
CREATE TABLE IF NOT EXISTS branches (
    id    INTEGER PRIMARY KEY,
    code  TEXT NOT NULL UNIQUE,   -- "CSE", "ECE", "MECH", ...
    name  TEXT NOT NULL
);

/* ── Students (master data, admin-only writes) ─────────────── */
CREATE TABLE IF NOT EXISTS students (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no         TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    email           TEXT    UNIQUE,
    phone           TEXT,
    branch          TEXT    NOT NULL,
    semester        INTEGER NOT NULL CHECK(semester BETWEEN 1 AND 8),
    section         TEXT    DEFAULT 'A',
    dob             TEXT,
    gender          TEXT    CHECK(gender IN ('M','F','Other')),
    address         TEXT,
    guardian_name   TEXT,
    guardian_phone  TEXT,
    photo_path      TEXT,
    embedding       BLOB,              -- 512-d float32 array
    face_enrolled   INTEGER DEFAULT 0,
    user_id         INTEGER UNIQUE REFERENCES users(id) ON DELETE SET NULL,
    enrolled_at     TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    active          INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_stu_branch_sem ON students(branch, semester);

/* ── Faculty (master data, admin-only writes) ──────────────── */
CREATE TABLE IF NOT EXISTS faculty (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id     TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    email           TEXT    UNIQUE,
    phone           TEXT,
    branch          TEXT    NOT NULL,
    designation     TEXT    DEFAULT 'Assistant Professor',
    qualification   TEXT,
    experience_yrs  INTEGER DEFAULT 0,
    photo_path      TEXT,
    embedding       BLOB,
    face_enrolled   INTEGER DEFAULT 0,
    user_id         INTEGER UNIQUE REFERENCES users(id) ON DELETE SET NULL,
    joined_at       TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    active          INTEGER DEFAULT 1
);

/* ── Courses ───────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS courses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    code            TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    branch          TEXT    NOT NULL,
    semester        INTEGER NOT NULL CHECK(semester BETWEEN 1 AND 8),
    credits         INTEGER DEFAULT 3,
    type            TEXT    DEFAULT 'Theory' CHECK(type IN ('Theory','Lab','Elective')),
    faculty_id      INTEGER REFERENCES faculty(id) ON DELETE SET NULL,
    academic_year   TEXT    DEFAULT '2024-25',
    active          INTEGER DEFAULT 1,
    created_at      TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);
CREATE INDEX IF NOT EXISTS idx_course_branch_sem ON courses(branch, semester);

/* ── Enrollment (many-to-many: students <-> courses) ───────── */
CREATE TABLE IF NOT EXISTS enrollments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    course_id  INTEGER NOT NULL REFERENCES courses(id)  ON DELETE CASCADE,
    enrolled_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    UNIQUE(student_id, course_id)
);
CREATE INDEX IF NOT EXISTS idx_enroll_stu ON enrollments(student_id);
CREATE INDEX IF NOT EXISTS idx_enroll_crs ON enrollments(course_id);

/* ── Attendance records ─────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS attendance (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id          INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    course_id           INTEGER NOT NULL REFERENCES courses(id)  ON DELETE CASCADE,
    date                TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d','now')),
    entry_time          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    exit_time           TEXT,
    status              TEXT    NOT NULL DEFAULT 'PRESENT'
                                    CHECK(status IN ('PRESENT','LATE','ABSENT')),
    recognition_confidence REAL DEFAULT 0.0,
    liveness_score      REAL    DEFAULT 0.0,
    event_type          TEXT    NOT NULL CHECK(event_type IN ('ENTRY','EXIT')),
    device_id           INTEGER REFERENCES devices(id),
    marked_by           TEXT    DEFAULT 'AUTO' CHECK(marked_by IN ('AUTO','MANUAL')),
    teacher_id          INTEGER REFERENCES faculty(id),
    notes               TEXT,
    UNIQUE(student_id, course_id, date, event_type)
);
CREATE INDEX IF NOT EXISTS idx_att_stu_date   ON attendance(student_id, date);
CREATE INDEX IF NOT EXISTS idx_att_crs_date   ON attendance(course_id, date);
CREATE INDEX IF NOT EXISTS idx_att_date       ON attendance(date);

/* ── Devices (cameras) ─────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS devices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    location    TEXT,
    camera_idx  INTEGER DEFAULT 0,
    status      TEXT    DEFAULT 'online',
    last_seen   TEXT,
    active      INTEGER DEFAULT 1
);

/* ── Auth users ────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL CHECK(role IN ('admin','teacher','student')),
    active        INTEGER DEFAULT 1,
    last_login    TEXT,
    created_at    TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);

/* ── Sessions ──────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT    PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL,
    entity_id   INTEGER,   -- student.id or faculty.id
    created_at  TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    expires_at  TEXT    NOT NULL
);

/* ── Audit log ─────────────────────────────────────────────── */
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id    INTEGER REFERENCES users(id),
    action      TEXT    NOT NULL,
    entity_type TEXT,
    entity_id   INTEGER,
    detail      TEXT,
    ts          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);
"""

BRANCHES_SEED = """
INSERT OR IGNORE INTO branches (code, name) VALUES
  ('CSE',  'Computer Science & Engineering'),
  ('ECE',  'Electronics & Communication Engineering'),
  ('MECH', 'Mechanical Engineering'),
  ('CIVIL','Civil Engineering'),
  ('EEE',  'Electrical & Electronics Engineering'),
  ('IT',   'Information Technology'),
  ('AIDS', 'AI & Data Science');
"""

DEFAULT_DEVICE = """
INSERT OR IGNORE INTO devices (id, name, location, camera_idx, status)
VALUES (1, 'Classroom Camera 1', 'Main Block', 0, 'online');
"""

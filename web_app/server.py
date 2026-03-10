"""
web_app/server.py
=================
Flask API — all endpoints for admin / teacher / student dashboards.

Admin:
  GET/POST/PUT/DELETE  /api/admin/students
  GET/POST/PUT/DELETE  /api/admin/faculty
  GET/POST/PUT/DELETE  /api/admin/courses
  GET                  /api/admin/attendance   (filterable)
  GET                  /api/admin/stats
  POST                 /api/admin/enroll_photo  (upload + embed)

Teacher:
  GET   /api/teacher/courses          (assigned to them)
  GET   /api/teacher/class_roster     (students in a course)
  GET   /api/teacher/attendance       (their course, date range)
  GET   /api/teacher/stats

Student:
  GET   /api/student/profile
  GET   /api/student/courses
  GET   /api/student/attendance

Common:
  POST  /api/auth/login
  POST  /api/auth/logout
  GET   /api/auth/me
  GET   /api/video_feed
  GET   /api/live/faces
  GET   /api/events/stream            (SSE)
"""

from __future__ import annotations
import base64, io, json, os, queue, sys, threading, time
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, stream_with_context)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auth.manager import Auth
from database.db import DB

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'sas-v2-secret-change-me'

# ── Injected globals ──────────────────────────────────────────────────────────
_db: Optional[DB]   = None
_auth: Optional[Auth] = None
_result_q: Optional[queue.Queue] = None
_latest_frame: Optional[np.ndarray] = None
_latest_faces: list = []
_frame_lock = threading.Lock()
_uploads_dir: Path  = Path('uploads')


def init_server(db: DB, result_q: queue.Queue, uploads_dir: Path):
    global _db, _auth, _result_q, _uploads_dir
    _db, _auth, _result_q = db, Auth(db), result_q
    _uploads_dir = uploads_dir
    _uploads_dir.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_frame_sink, daemon=True).start()


def _frame_sink():
    global _latest_frame, _latest_faces
    while True:
        try:
            res = _result_q.get(timeout=1)
            with _frame_lock:
                _latest_frame = res.frame
                _latest_faces = res.faces
        except Exception:
            pass


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _tok():
    return (request.cookies.get('sas_tok')
            or request.headers.get('X-Auth-Token')
            or request.args.get('token'))


def require(*roles):
    def dec(f):
        @wraps(f)
        def wrap(*a, **kw):
            s = _auth.require(_tok(), *roles) if _auth else None
            if not s:
                return (jsonify({'error': 'Unauthorized'}), 401) if request.path.startswith('/api/') else redirect('/login')
            request.sess = s
            return f(*a, **kw)
        return wrap
    return dec


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route('/')
def root():
    s = _auth.session(_tok()) if _auth else None
    return redirect(f"/{s['role']}" if s else '/login')

@app.route('/login')
def login_pg(): return render_template('login.html')

@app.route('/admin')
@require('admin')
def admin_pg(): return render_template('admin.html', sess=request.sess)

@app.route('/teacher')
@require('teacher', 'admin')
def teacher_pg(): return render_template('teacher.html', sess=request.sess)

@app.route('/student')
@require('student', 'admin')
def student_pg(): return render_template('student.html', sess=request.sess)


# ── Auth API ──────────────────────────────────────────────────────────────────

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    d = request.get_json() or {}
    s = _auth.login(d.get('email',''), d.get('password',''), request.remote_addr)
    if not s: return jsonify({'error': 'Invalid email or password'}), 401
    r = jsonify({'role': s['role'], 'redirect': f"/{s['role']}"})
    r.set_cookie('sas_tok', s['token'], httponly=True, samesite='Lax', max_age=8*3600)
    return r

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    if _auth: _auth.logout(_tok())
    r = jsonify({'ok': True}); r.delete_cookie('sas_tok'); return r

@app.route('/api/auth/me')
def api_me():
    s = _auth.session(_tok()) if _auth else None
    return jsonify(s) if s else (jsonify({'error': 'Not authenticated'}), 401)


# ── Video feed ────────────────────────────────────────────────────────────────

def _mjpeg():
    no_frame_count = 0
    while True:
        with _frame_lock: frame = _latest_frame
        if frame is None:
            no_frame_count += 1
            frame = np.zeros((360,640,3), np.uint8)
            if no_frame_count < 50:
                cv2.putText(frame, 'Camera initialising...', (150,180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 2)
            else:
                cv2.putText(frame, 'Camera not available', (120,170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100,100,255), 2)
                cv2.putText(frame, 'Check: drivers, permissions, USB', (40,220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255), 1)
        else:
            no_frame_count = 0
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'
        time.sleep(1/25)

@app.route('/api/video_feed')
def video_feed():
    return Response(stream_with_context(_mjpeg()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/live/faces')
@require('admin','teacher')
def live_faces():
    with _frame_lock:
        return jsonify([{'track_id': f.track_id, 'name': f.name,
                         'confidence': f.identity_conf,
                         'liveness': f.liveness_passed,
                         'event': f.event_type} for f in _latest_faces])


# ════════════════════════════════════════════════════════════════════════
#  ADMIN APIs
# ════════════════════════════════════════════════════════════════════════

# ── Admin stats ──────────────────────────────────────────────────────────────

@app.route('/api/admin/stats')
@require('admin')
def admin_stats():
    today = date.today().isoformat()
    return jsonify({
        'total_students': _db.one("SELECT COUNT(*) n FROM students WHERE active=1")['n'],
        'total_faculty':  _db.one("SELECT COUNT(*) n FROM faculty  WHERE active=1")['n'],
        'total_courses':  _db.one("SELECT COUNT(*) n FROM courses  WHERE active=1")['n'],
        'present_today':  _db.one("SELECT COUNT(DISTINCT student_id) n FROM attendance WHERE date=? AND event_type='ENTRY'", (today,))['n'],
    })


# ── Admin: Students ──────────────────────────────────────────────────────────

@app.route('/api/admin/students', methods=['GET'])
@require('admin')
def get_students():
    branch = request.args.get('branch','')
    sem    = request.args.get('semester','')
    q = ("SELECT s.*, "
         "(SELECT COUNT(*) FROM enrollments e WHERE e.student_id=s.id) AS enrolled_courses "
         "FROM students s WHERE s.active=1")
    p = []
    if branch: q += " AND s.branch=?"; p.append(branch)
    if sem:    q += " AND s.semester=?"; p.append(int(sem))
    q += " ORDER BY s.branch, s.semester, s.roll_no"
    return jsonify(_db.all(q, p))


@app.route('/api/admin/students', methods=['POST'])
@require('admin')
def create_student():
    d = request.get_json() or {}
    required = ('roll_no','name','branch','semester')
    if not all(d.get(k) for k in required):
        return jsonify({'error': 'roll_no, name, branch, semester required'}), 400

    # Create auth user if email provided
    user_id = None
    if d.get('email'):
        try:
            pw = d.get('password', d['roll_no'])  # default pw = roll_no
            user_id = _auth.create_user(d['email'], pw, 'student')
        except Exception as e:
            return jsonify({'error': f'Email already exists: {e}'}), 400

    with _db.tx() as c:
        cur = c.execute(
            "INSERT INTO students (roll_no,name,email,phone,branch,semester,section,"
            "dob,gender,address,guardian_name,guardian_phone,user_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (d['roll_no'], d['name'], d.get('email'), d.get('phone'),
             d['branch'], int(d['semester']), d.get('section','A'),
             d.get('dob'), d.get('gender'), d.get('address'),
             d.get('guardian_name'), d.get('guardian_phone'), user_id)
        )
        stu_id = cur.lastrowid
        _audit('admin', 'CREATE', 'student', stu_id, f"Roll: {d['roll_no']}")

    # Auto-enroll in all courses matching branch+semester
    _auto_enroll(stu_id, d['branch'], int(d['semester']))
    return jsonify({'id': stu_id, 'ok': True})


@app.route('/api/admin/students/<int:sid>', methods=['PUT'])
@require('admin')
def update_student(sid):
    d = request.get_json() or {}
    fields = {k: d[k] for k in ('name','phone','branch','semester','section',
                                  'dob','gender','address','guardian_name','guardian_phone')
              if k in d}
    if not fields: return jsonify({'error': 'Nothing to update'}), 400
    set_clause = ', '.join(f"{k}=?" for k in fields)
    with _db.tx() as c:
        c.execute(f"UPDATE students SET {set_clause} WHERE id=?",
                  list(fields.values()) + [sid])
        _audit('admin', 'UPDATE', 'student', sid, str(fields))
    return jsonify({'ok': True})


@app.route('/api/admin/students/<int:sid>', methods=['DELETE'])
@require('admin')
def delete_student(sid):
    with _db.tx() as c:
        c.execute("UPDATE students SET active=0 WHERE id=?", (sid,))
        _audit('admin', 'DELETE', 'student', sid, '')
    return jsonify({'ok': True})


@app.route('/api/admin/students/<int:sid>/photo', methods=['POST'])
@require('admin')
def upload_student_photo(sid):
    return _upload_photo('student', sid)


# ── Admin: Faculty ────────────────────────────────────────────────────────────

@app.route('/api/admin/faculty', methods=['GET'])
@require('admin')
def get_faculty():
    branch = request.args.get('branch','')
    q = ("SELECT f.*, "
         "(SELECT COUNT(*) FROM courses c WHERE c.faculty_id=f.id AND c.active=1) AS course_count "
         "FROM faculty f WHERE f.active=1")
    p = []
    if branch: q += " AND f.branch=?"; p.append(branch)
    q += " ORDER BY f.branch, f.name"
    return jsonify(_db.all(q, p))


@app.route('/api/admin/faculty', methods=['POST'])
@require('admin')
def create_faculty():
    d = request.get_json() or {}
    required = ('employee_id','name','branch')
    if not all(d.get(k) for k in required):
        return jsonify({'error': 'employee_id, name, branch required'}), 400

    user_id = None
    if d.get('email'):
        try:
            pw = d.get('password', d['employee_id'])
            user_id = _auth.create_user(d['email'], pw, 'teacher')
        except Exception as e:
            return jsonify({'error': f'Email exists: {e}'}), 400

    with _db.tx() as c:
        cur = c.execute(
            "INSERT INTO faculty (employee_id,name,email,phone,branch,designation,"
            "qualification,experience_yrs,user_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (d['employee_id'], d['name'], d.get('email'), d.get('phone'),
             d['branch'], d.get('designation','Assistant Professor'),
             d.get('qualification'), d.get('experience_yrs', 0), user_id)
        )
        fac_id = cur.lastrowid
        _audit('admin', 'CREATE', 'faculty', fac_id, f"EmpID: {d['employee_id']}")
    return jsonify({'id': fac_id, 'ok': True})


@app.route('/api/admin/faculty/<int:fid>', methods=['PUT'])
@require('admin')
def update_faculty(fid):
    d = request.get_json() or {}
    fields = {k: d[k] for k in ('name','phone','branch','designation',
                                  'qualification','experience_yrs') if k in d}
    if not fields: return jsonify({'error': 'Nothing to update'}), 400
    set_clause = ', '.join(f"{k}=?" for k in fields)
    with _db.tx() as c:
        c.execute(f"UPDATE faculty SET {set_clause} WHERE id=?",
                  list(fields.values()) + [fid])
    return jsonify({'ok': True})


@app.route('/api/admin/faculty/<int:fid>', methods=['DELETE'])
@require('admin')
def delete_faculty(fid):
    with _db.tx() as c:
        c.execute("UPDATE faculty SET active=0 WHERE id=?", (fid,))
        _audit('admin', 'DELETE', 'faculty', fid, '')
    return jsonify({'ok': True})


@app.route('/api/admin/faculty/<int:fid>/photo', methods=['POST'])
@require('admin')
def upload_faculty_photo(fid):
    return _upload_photo('faculty', fid)


# ── Admin: Courses ────────────────────────────────────────────────────────────

@app.route('/api/admin/courses', methods=['GET'])
@require('admin','teacher')
def get_courses():
    branch = request.args.get('branch','')
    sem    = request.args.get('semester','')
    fac    = request.args.get('faculty_id','')
    q = ("SELECT c.*, f.name AS faculty_name, f.employee_id, "
         "(SELECT COUNT(*) FROM enrollments e WHERE e.course_id=c.id) AS enrolled_count "
         "FROM courses c LEFT JOIN faculty f ON f.id=c.faculty_id WHERE c.active=1")
    p = []
    if branch: q += " AND c.branch=?"; p.append(branch)
    if sem:    q += " AND c.semester=?"; p.append(int(sem))
    if fac:    q += " AND c.faculty_id=?"; p.append(int(fac))
    q += " ORDER BY c.branch, c.semester, c.code"
    return jsonify(_db.all(q, p))


@app.route('/api/admin/courses', methods=['POST'])
@require('admin')
def create_course():
    d = request.get_json() or {}
    required = ('code','name','branch','semester')
    if not all(d.get(k) for k in required):
        return jsonify({'error': 'code, name, branch, semester required'}), 400
    with _db.tx() as c:
        cur = c.execute(
            "INSERT INTO courses (code,name,branch,semester,credits,type,faculty_id,academic_year) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (d['code'], d['name'], d['branch'], int(d['semester']),
             d.get('credits',3), d.get('type','Theory'),
             d.get('faculty_id') or None, d.get('academic_year','2024-25'))
        )
        cid = cur.lastrowid
        _audit('admin', 'CREATE', 'course', cid, f"Code: {d['code']}")
    # Auto-enroll matching students
    _auto_enroll_course(cid, d['branch'], int(d['semester']))
    return jsonify({'id': cid, 'ok': True})


@app.route('/api/admin/courses/<int:cid>', methods=['PUT'])
@require('admin')
def update_course(cid):
    d = request.get_json() or {}
    fields = {k: d[k] for k in ('name','branch','semester','credits','type',
                                  'faculty_id','academic_year') if k in d}
    if not fields: return jsonify({'error': 'Nothing to update'}), 400
    set_clause = ', '.join(f"{k}=?" for k in fields)
    with _db.tx() as c:
        c.execute(f"UPDATE courses SET {set_clause} WHERE id=?",
                  list(fields.values()) + [cid])
    return jsonify({'ok': True})


@app.route('/api/admin/courses/<int:cid>', methods=['DELETE'])
@require('admin')
def delete_course(cid):
    with _db.tx() as c:
        c.execute("UPDATE courses SET active=0 WHERE id=?", (cid,))
    return jsonify({'ok': True})


# ── Admin: Attendance sheets ──────────────────────────────────────────────────

@app.route('/api/admin/attendance')
@require('admin')
def admin_attendance():
    d_from = request.args.get('from', date.today().isoformat())
    d_to   = request.args.get('to',   date.today().isoformat())
    branch = request.args.get('branch','')
    sem    = request.args.get('semester','')
    course = request.args.get('course_id','')
    student= request.args.get('student_id','')

    q = """
        SELECT a.*, s.name AS student_name, s.roll_no, s.branch, s.semester,
               c.name AS course_name, c.code AS course_code,
               f.name AS teacher_name
        FROM attendance a
        JOIN students s ON s.id=a.student_id
        JOIN courses  c ON c.id=a.course_id
        LEFT JOIN faculty f ON f.id=c.faculty_id
        WHERE a.date BETWEEN ? AND ?
    """
    p = [d_from, d_to]
    if branch:  q += " AND s.branch=?";   p.append(branch)
    if sem:     q += " AND s.semester=?"; p.append(int(sem))
    if course:  q += " AND a.course_id=?";p.append(int(course))
    if student: q += " AND a.student_id=?";p.append(int(student))
    q += " ORDER BY a.date DESC, a.entry_time DESC LIMIT 500"
    return jsonify(_db.all(q, p))


@app.route('/api/admin/attendance/summary')
@require('admin')
def admin_att_summary():
    """Class-level summary: course × date → present count."""
    today = request.args.get('date', date.today().isoformat())
    rows = _db.all("""
        SELECT c.name AS course_name, c.code, c.branch, c.semester,
               COUNT(DISTINCT e.student_id) AS total,
               COUNT(DISTINCT CASE WHEN a.date=? AND a.event_type='ENTRY'
                             THEN a.student_id END) AS present
        FROM courses c
        LEFT JOIN enrollments e ON e.course_id=c.id
        LEFT JOIN attendance  a ON a.course_id=c.id AND a.student_id=e.student_id
        WHERE c.active=1
        GROUP BY c.id ORDER BY c.branch, c.semester, c.code
    """, (today,))
    return jsonify(rows)


# ── Admin: Analytics ──────────────────────────────────────────────────────────

@app.route('/api/admin/analytics/trend')
@require('admin')
def admin_trend():
    days = int(request.args.get('days', 14))
    rows = []
    for i in range(days-1, -1, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        n = _db.one("SELECT COUNT(DISTINCT student_id) n FROM attendance WHERE date=? AND event_type='ENTRY'", (d,))['n']
        rows.append({'date': d, 'count': n})
    return jsonify(rows)


@app.route('/api/admin/analytics/branch_summary')
@require('admin')
def branch_summary():
    today = date.today().isoformat()
    return jsonify(_db.all("""
        SELECT s.branch,
               COUNT(DISTINCT s.id) AS total_students,
               COUNT(DISTINCT CASE WHEN a.date=? AND a.event_type='ENTRY' THEN a.student_id END) AS present
        FROM students s
        LEFT JOIN attendance a ON a.student_id=s.id
        WHERE s.active=1 GROUP BY s.branch ORDER BY total_students DESC
    """, (today,)))


@app.route('/api/admin/audit')
@require('admin')
def admin_audit():
    return jsonify(_db.all("SELECT * FROM audit_log ORDER BY ts DESC LIMIT 100"))


@app.route('/api/admin/branches')
@require('admin','teacher','student')
def get_branches():
    return jsonify(_db.all("SELECT * FROM branches ORDER BY code"))


# ════════════════════════════════════════════════════════════════════════
#  TEACHER APIs
# ════════════════════════════════════════════════════════════════════════

@app.route('/api/teacher/courses')
@require('teacher','admin')
def teacher_courses():
    fac_id = request.sess.get('entity_id')
    if not fac_id:
        return jsonify([])
    return jsonify(_db.all("""
        SELECT c.*,
               (SELECT COUNT(*) FROM enrollments e WHERE e.course_id=c.id) AS total_students
        FROM courses c WHERE c.faculty_id=? AND c.active=1
        ORDER BY c.branch, c.semester, c.name
    """, (fac_id,)))


@app.route('/api/teacher/roster')
@require('teacher','admin')
def teacher_roster():
    cid   = request.args.get('course_id')
    d_str = request.args.get('date', date.today().isoformat())
    if not cid: return jsonify({'error': 'course_id required'}), 400

    students = _db.all("""
        SELECT s.id, s.roll_no, s.name, s.branch, s.semester, s.photo_path,
               a_entry.entry_time,
               a_entry.recognition_confidence,
               a_entry.liveness_score,
               a_entry.status,
               a_entry.marked_by
        FROM enrollments e
        JOIN students s ON s.id=e.student_id
        LEFT JOIN attendance a_entry ON a_entry.student_id=s.id
                  AND a_entry.course_id=? AND a_entry.date=? AND a_entry.event_type='ENTRY'
        WHERE e.course_id=? AND s.active=1
        ORDER BY s.roll_no
    """, (cid, d_str, cid))
    return jsonify(students)


@app.route('/api/teacher/attendance')
@require('teacher','admin')
def teacher_attendance():
    fac_id = request.sess.get('entity_id')
    cid    = request.args.get('course_id')
    d_from = request.args.get('from', date.today().isoformat())
    d_to   = request.args.get('to',   date.today().isoformat())

    q = """
        SELECT a.*, s.name AS student_name, s.roll_no,
               c.name AS course_name, c.code AS course_code
        FROM attendance a
        JOIN students s ON s.id=a.student_id
        JOIN courses  c ON c.id=a.course_id
        WHERE c.faculty_id=? AND a.date BETWEEN ? AND ?
    """
    p = [fac_id, d_from, d_to]
    if cid: q += " AND a.course_id=?"; p.append(int(cid))
    q += " ORDER BY a.date DESC, a.entry_time DESC LIMIT 500"
    return jsonify(_db.all(q, p))


@app.route('/api/teacher/stats')
@require('teacher','admin')
def teacher_stats():
    fac_id = request.sess.get('entity_id')
    today  = date.today().isoformat()
    return jsonify({
        'total_courses':  _db.one("SELECT COUNT(*) n FROM courses WHERE faculty_id=? AND active=1", (fac_id,))['n'],
        'total_students': _db.one("SELECT COUNT(DISTINCT e.student_id) n FROM enrollments e JOIN courses c ON c.id=e.course_id WHERE c.faculty_id=?", (fac_id,))['n'],
        'present_today':  _db.one("SELECT COUNT(DISTINCT a.student_id) n FROM attendance a JOIN courses c ON c.id=a.course_id WHERE c.faculty_id=? AND a.date=? AND a.event_type='ENTRY'", (fac_id, today))['n'],
    })


@app.route('/api/teacher/manual_mark', methods=['POST'])
@require('teacher','admin')
def teacher_manual_mark():
    """Teacher manually marks a student present/absent."""
    d      = request.get_json() or {}
    fac_id = request.sess.get('entity_id')
    today  = date.today().isoformat()
    cid    = d.get('course_id')
    stu_id = d.get('student_id')
    status = d.get('status', 'PRESENT')
    notes  = d.get('notes', '')

    if not cid or not stu_id:
        return jsonify({'error': 'course_id and student_id required'}), 400

    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with _db.tx() as c:
        c.execute("""
            INSERT INTO attendance (student_id,course_id,date,entry_time,status,
                                    event_type,recognition_confidence,liveness_score,
                                    marked_by,teacher_id,notes)
            VALUES (?,?,?,?,'PRESENT','ENTRY',1.0,1.0,'MANUAL',?,?)
            ON CONFLICT(student_id,course_id,date,event_type) DO UPDATE SET
                status=excluded.status, marked_by='MANUAL', notes=excluded.notes
        """, (stu_id, cid, today, now, fac_id, notes))
    return jsonify({'ok': True})


@app.route('/api/teacher/analytics/trend')
@require('teacher','admin')
def teacher_trend():
    """Teacher attendance trend for their courses."""
    days = int(request.args.get('days', 14))
    fac_id = request.sess.get('entity_id')
    rows = []
    for i in range(days-1, -1, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        n = _db.one("""
            SELECT COUNT(DISTINCT a.student_id) n FROM attendance a
            JOIN courses c ON c.id=a.course_id
            WHERE c.faculty_id=? AND a.date=? AND a.event_type='ENTRY'
        """, (fac_id, d))['n']
        rows.append({'date': d, 'count': n})
    return jsonify(rows)


# ════════════════════════════════════════════════════════════════════════
#  STUDENT APIs
# ════════════════════════════════════════════════════════════════════════

@app.route('/api/student/profile')
@require('student','admin')
def student_profile():
    stu_id = request.sess.get('entity_id')
    s = _db.one("SELECT * FROM students WHERE id=?", (stu_id,))
    if s: s.pop('embedding', None)  # don't send binary
    return jsonify(s)


@app.route('/api/student/courses')
@require('student','admin')
def student_courses():
    stu_id = request.sess.get('entity_id')
    return jsonify(_db.all("""
        SELECT c.*, f.name AS faculty_name,
               (SELECT COUNT(*) FROM attendance a
                WHERE a.student_id=? AND a.course_id=c.id AND a.event_type='ENTRY') AS attended,
               (SELECT COUNT(DISTINCT a2.date) FROM attendance a2
                WHERE a2.course_id=c.id AND a2.event_type='ENTRY') AS total_classes
        FROM enrollments e
        JOIN courses c ON c.id=e.course_id
        LEFT JOIN faculty f ON f.id=c.faculty_id
        WHERE e.student_id=? AND c.active=1
        ORDER BY c.semester, c.name
    """, (stu_id, stu_id)))


@app.route('/api/student/attendance')
@require('student','admin')
def student_attendance():
    stu_id = request.sess.get('entity_id')
    cid    = request.args.get('course_id')
    d_from = request.args.get('from', (date.today()-timedelta(days=30)).isoformat())
    d_to   = request.args.get('to',   date.today().isoformat())
    q = """
        SELECT a.*, c.name AS course_name, c.code AS course_code
        FROM attendance a
        JOIN courses c ON c.id=a.course_id
        WHERE a.student_id=? AND a.date BETWEEN ? AND ?
    """
    p = [stu_id, d_from, d_to]
    if cid: q += " AND a.course_id=?"; p.append(int(cid))
    q += " ORDER BY a.date DESC, a.entry_time DESC"
    return jsonify(_db.all(q, p))


@app.route('/api/student/summary')
@require('student','admin')
def student_summary():
    stu_id = request.sess.get('entity_id')
    today  = date.today().isoformat()
    today_row = _db.one("SELECT event_type FROM attendance WHERE student_id=? AND date=? AND event_type='ENTRY' LIMIT 1", (stu_id, today))
    return jsonify({
        'today_status': 'PRESENT' if today_row else 'ABSENT',
        'total_courses': _db.one("SELECT COUNT(*) n FROM enrollments e JOIN courses c ON c.id=e.course_id WHERE e.student_id=? AND c.active=1", (stu_id,))['n'],
    })


# ════════════════════════════════════════════════════════════════════════
#  SSE
# ════════════════════════════════════════════════════════════════════════

_sse_qs: list = []
_sse_lock = threading.Lock()


def sse_push(data: dict):
    msg = f"data: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_qs:
            try: q.put_nowait(msg)
            except Exception: dead.append(q)
        for q in dead: _sse_qs.remove(q)


@app.route('/api/events/stream')
@require('admin','teacher','student')
def sse_stream():
    def gen():
        q: queue.Queue = queue.Queue(20)
        with _sse_lock: _sse_qs.append(q)
        try:
            yield 'data: {"type":"connected"}\n\n'
            while True:
                try: yield q.get(timeout=25)
                except queue.Empty: yield ': ping\n\n'
        finally:
            with _sse_lock:
                if q in _sse_qs: _sse_qs.remove(q)
    return Response(stream_with_context(gen()), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})


# ── Uploads ────────────────────────────────────────────────────────────────────

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    from flask import send_from_directory
    return send_from_directory(str(_uploads_dir), filename)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _upload_photo(entity_type: str, entity_id: int):
    """Handle photo upload + face embedding generation."""
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo file'}), 400
    file = request.files['photo']
    ext  = Path(file.filename).suffix.lower() or '.jpg'
    fname = f"{entity_type}_{entity_id}{ext}"
    fpath = _uploads_dir / fname
    file.save(str(fpath))

    # Generate face embedding from photo
    embedding_blob = None
    try:
        img = cv2.imread(str(fpath))
        if img is not None:
            embedding_blob = _generate_embedding_from_image(img)
    except Exception as e:
        print(f"Embedding error: {e}")

    table = 'students' if entity_type == 'student' else 'faculty'
    with _db.tx() as c:
        if embedding_blob is not None:
            c.execute(f"UPDATE {table} SET photo_path=?, embedding=?, face_enrolled=1 WHERE id=?",
                      (fname, embedding_blob, entity_id))
            # Reload face index
            _reload_face_index()
        else:
            c.execute(f"UPDATE {table} SET photo_path=? WHERE id=?", (fname, entity_id))

    _audit('admin', 'PHOTO_UPLOAD', entity_type, entity_id, fname)
    return jsonify({'ok': True, 'enrolled': embedding_blob is not None, 'path': fname})


def _generate_embedding_from_image(img: np.ndarray):
    """Try to extract a face embedding from a still image."""
    global _detector_ref, _embedder_ref, _aligner_ref
    try:
        if _detector_ref and _embedder_ref and _aligner_ref:
            dets = _detector_ref.detect(img)
            if dets:
                best = max(dets, key=lambda d: d.area)
                aligned = _aligner_ref.align(img, best.bbox, best.landmarks)
                if aligned is not None:
                    emb = _embedder_ref.get_embedding(aligned)
                    if emb is not None:
                        return _db.embed_to_blob(emb)
    except Exception as e:
        print(f"_generate_embedding: {e}")
    return None


def _reload_face_index():
    """Reload face embeddings into the FAISS searcher after photo upload."""
    global _searcher_ref
    try:
        if _searcher_ref:
            records = []
            for stu in _db.all("SELECT id, name, branch, embedding FROM students WHERE active=1 AND face_enrolled=1"):
                if stu['embedding']:
                    records.append({'user_id': stu['id'], 'name': stu['name'],
                                    'department': stu['branch'],
                                    'embedding': _db.blob_to_embed(stu['embedding'])})
            _searcher_ref.add_faces_bulk(records)
    except Exception as e:
        print(f"reload_face_index: {e}")


def _auto_enroll(stu_id: int, branch: str, semester: int):
    """Auto-enroll a new student in all active courses for their branch+semester."""
    courses = _db.all("SELECT id FROM courses WHERE branch=? AND semester=? AND active=1",
                      (branch, semester))
    with _db.tx() as c:
        for co in courses:
            c.execute("INSERT OR IGNORE INTO enrollments (student_id,course_id) VALUES (?,?)",
                      (stu_id, co['id']))


def _auto_enroll_course(course_id: int, branch: str, semester: int):
    """Auto-enroll all matching students when a new course is created."""
    students = _db.all("SELECT id FROM students WHERE branch=? AND semester=? AND active=1",
                       (branch, semester))
    with _db.tx() as c:
        for s in students:
            c.execute("INSERT OR IGNORE INTO enrollments (student_id,course_id) VALUES (?,?)",
                      (s['id'], course_id))


def _audit(actor: str, action: str, entity_type: str, entity_id: int, detail: str):
    _db.run("INSERT INTO audit_log (actor_id,action,entity_type,entity_id,detail) VALUES (1,?,?,?,?)",
            (action, entity_type, entity_id, detail))


# ── ML component references (injected by main) ────────────────────────────────
_detector_ref  = None
_embedder_ref  = None
_aligner_ref   = None
_searcher_ref  = None


def inject_ml(detector, embedder, aligner, searcher):
    global _detector_ref, _embedder_ref, _aligner_ref, _searcher_ref
    _detector_ref = detector
    _embedder_ref = embedder
    _aligner_ref  = aligner
    _searcher_ref = searcher


# ── Export reports (inline, no extra module needed) ──────────────────────────
@app.route('/api/reports/export')
def export_report():
    tok  = _tok()
    sess = _auth.require(tok, 'admin', 'teacher') if _auth else None
    if not sess:
        return jsonify({'error': 'Unauthorized'}), 401

    import csv
    from datetime import date
    fmt     = request.args.get('format', 'csv')
    today_s = date.today().isoformat()
    outdir  = Path(__file__).parent.parent / 'exports'
    outdir.mkdir(parents=True, exist_ok=True)
    fname   = outdir / f'attendance_{today_s}.csv'

    records = _db.all("""
        SELECT a.date, s.roll_no, s.name AS student_name, s.branch, s.semester,
               c.code AS course_code, c.name AS course_name,
               a.entry_time, a.exit_time, a.status,
               a.recognition_confidence, a.liveness_score, a.marked_by
        FROM attendance a
        JOIN students s ON s.id=a.student_id
        JOIN courses  c ON c.id=a.course_id
        WHERE a.date=? ORDER BY a.entry_time
    """, (today_s,))

    with open(str(fname), 'w', newline='') as f:
        if records:
            w = csv.DictWriter(f, fieldnames=records[0].keys())
            w.writeheader(); w.writerows(records)

    return jsonify({'ok': True, 'path': str(fname)})

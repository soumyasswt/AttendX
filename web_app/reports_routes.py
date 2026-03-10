"""
web_app/reports_routes.py
=========================
Export routes injected into server.py
"""
from datetime import date
from pathlib import Path

import csv, io

def register_report_routes(app, db_getter, auth_getter, exports_dir: Path):
    """Call this from init_server to register export routes."""
    from flask import jsonify, request, send_file

    def _tok():
        from flask import request
        return (request.cookies.get('sas_tok')
                or request.headers.get('X-Auth-Token')
                or request.args.get('token'))

    @app.route('/api/reports/export')
    def export_report():
        _db   = db_getter()
        _auth = auth_getter()
        tok   = _tok()
        sess  = _auth.require(tok, 'admin', 'teacher')
        if not sess:
            return jsonify({'error': 'Unauthorized'}), 401

        fmt     = request.args.get('format', 'csv')
        today_s = date.today().isoformat()
        exports_dir.mkdir(parents=True, exist_ok=True)
        fname   = exports_dir / f'attendance_{today_s}.{fmt}'

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

        if fmt == 'csv':
            with open(str(fname), 'w', newline='') as f:
                if records:
                    w = csv.DictWriter(f, fieldnames=records[0].keys())
                    w.writeheader(); w.writerows(records)
        else:
            # Simple CSV even for "excel" format
            fname = exports_dir / f'attendance_{today_s}.csv'
            with open(str(fname), 'w', newline='') as f:
                if records:
                    w = csv.DictWriter(f, fieldnames=records[0].keys())
                    w.writeheader(); w.writerows(records)

        return jsonify({'ok': True, 'path': str(fname)})

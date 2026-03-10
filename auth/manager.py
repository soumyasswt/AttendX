"""auth/manager.py"""
import hashlib, hmac, secrets
from datetime import datetime, timedelta
from typing import Optional
from database.db import DB


class Auth:
    TTL = 8  # hours

    def __init__(self, db: DB):
        self.db = db

    @staticmethod
    def hash_pw(pw: str) -> str:
        salt = secrets.token_hex(16)
        key  = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 260_000)
        return f"{salt}${key.hex()}"

    @staticmethod
    def verify_pw(pw: str, stored: str) -> bool:
        try:
            salt, key_hex = stored.split('$', 1)
            expected = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 260_000)
            return hmac.compare_digest(expected.hex(), key_hex)
        except Exception:
            return False

    def login(self, email: str, pw: str, ip: str = '') -> Optional[dict]:
        user = self.db.one(
            "SELECT u.*, "
            "  (SELECT s.id FROM students s WHERE s.user_id=u.id) AS student_entity_id,"
            "  (SELECT f.id FROM faculty  f WHERE f.user_id=u.id) AS faculty_entity_id "
            "FROM users u WHERE u.email=? AND u.active=1",
            (email.lower().strip(),)
        )
        if not user or not self.verify_pw(pw, user['password_hash']):
            return None

        entity_id = user['student_entity_id'] or user['faculty_entity_id']
        token     = secrets.token_urlsafe(32)
        expires   = (datetime.now() + timedelta(hours=self.TTL)).strftime('%Y-%m-%dT%H:%M:%S')

        with self.db.tx() as c:
            c.execute(
                "INSERT INTO sessions (token,user_id,role,entity_id,expires_at) VALUES (?,?,?,?,?)",
                (token, user['id'], user['role'], entity_id, expires)
            )
            c.execute("UPDATE users SET last_login=? WHERE id=?",
                      (datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), user['id']))
        return {'token': token, 'user_id': user['id'], 'role': user['role'],
                'entity_id': entity_id, 'expires_at': expires}

    def session(self, token: str) -> Optional[dict]:
        if not token: return None
        now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        row = self.db.one(
            "SELECT s.*, u.email FROM sessions s JOIN users u ON u.id=s.user_id "
            "WHERE s.token=? AND s.expires_at>?", (token, now)
        )
        if not row: return None
        # Attach name
        if row['role'] == 'student':
            stu = self.db.one("SELECT name, roll_no, branch, semester FROM students WHERE id=?",
                              (row['entity_id'],))
            row.update(stu or {})
        elif row['role'] == 'teacher':
            fac = self.db.one("SELECT name, employee_id, branch FROM faculty WHERE id=?",
                              (row['entity_id'],))
            row.update(fac or {})
        elif row['role'] == 'admin':
            row['name'] = 'Administrator'
        return row

    def logout(self, token: str):
        self.db.run("DELETE FROM sessions WHERE token=?", (token,))

    def create_user(self, email: str, pw: str, role: str) -> int:
        h = self.hash_pw(pw)
        with self.db.tx() as c:
            cur = c.execute("INSERT INTO users (email,password_hash,role) VALUES (?,?,?)",
                            (email.lower(), h, role))
            return cur.lastrowid

    def require(self, token: str, *roles) -> Optional[dict]:
        s = self.session(token)
        return s if s and s['role'] in roles else None

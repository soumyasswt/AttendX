"""
analytics/reports.py
====================
Analytics layer: generates summaries, statistics, and exportable reports.

All analytics are computed from the database via SQL aggregations.
Reports can be exported to CSV or Excel.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config.settings import AppConfig
from database.db import DB


class AttendanceAnalytics:
    """
    Generates structured analytics and reports from the attendance database.
    """

    def __init__(self, cfg: AppConfig, db: DB) -> None:
        self._cfg = cfg
        self._db = db

    # ------------------------------------------------------------------ #
    # Real-time / live stats                                               #
    # ------------------------------------------------------------------ #

    def get_live_stats(self) -> Dict:
        """
        Snapshot statistics for the dashboard header bar.
        Returned dict keys: total_enrolled, present_today, absent_today,
                            late_today, avg_entry_time
        """
        today = date.today().isoformat()

        total_enrolled = self._db.fetchone(
            "SELECT COUNT(*) AS n FROM users WHERE active = 1"
        )["n"]

        present_today = self._db.fetchone(
            """
            SELECT COUNT(DISTINCT user_id) AS n FROM attendance
            WHERE date = ? AND event_type = 'ENTRY'
            """,
            (today,),
        )["n"]

        late_today = self._db.fetchone(
            """
            SELECT COUNT(DISTINCT user_id) AS n FROM attendance
            WHERE date = ? AND status = 'LATE'
            """,
            (today,),
        )["n"]

        avg_entry_row = self._db.fetchone(
            """
            SELECT AVG(
                CAST(strftime('%H', entry_time) AS INTEGER) * 60 +
                CAST(strftime('%M', entry_time) AS INTEGER)
            ) AS avg_min
            FROM attendance
            WHERE date = ? AND event_type = 'ENTRY'
            """,
            (today,),
        )
        avg_min = avg_entry_row["avg_min"]
        if avg_min:
            avg_entry_time = f"{int(avg_min // 60):02d}:{int(avg_min % 60):02d}"
        else:
            avg_entry_time = "N/A"

        return {
            "total_enrolled": total_enrolled,
            "present_today": present_today,
            "absent_today": max(0, total_enrolled - present_today),
            "late_today": late_today,
            "avg_entry_time": avg_entry_time,
            "date": today,
        }

    # ------------------------------------------------------------------ #
    # Daily report                                                         #
    # ------------------------------------------------------------------ #

    def daily_report(self, report_date: Optional[str] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per user showing their attendance
        for the given date (default: today).

        Columns: Name, Department, Status, Entry Time, Exit Time,
                 Hours Worked, Late
        """
        if report_date is None:
            report_date = date.today().isoformat()

        rows = self._db.fetchall(
            """
            SELECT
                u.name, u.department,
                MIN(CASE WHEN a.event_type='ENTRY' THEN a.entry_time END) AS entry_time,
                MAX(CASE WHEN a.event_type='EXIT'  THEN a.entry_time END) AS exit_time,
                MAX(CASE WHEN a.status='LATE' THEN 1 ELSE 0 END)         AS was_late
            FROM attendance a
            JOIN users u ON u.id = a.user_id
            WHERE a.date = ?
            GROUP BY a.user_id
            ORDER BY entry_time
            """,
            (report_date,),
        )

        # All enrolled users (to detect absences)
        all_users = self._db.fetchall(
            "SELECT name, department FROM users WHERE active = 1 ORDER BY name"
        )

        present_names = {r["name"] for r in rows}
        records = []

        for row in rows:
            entry_dt = _parse_dt(row["entry_time"])
            exit_dt = _parse_dt(row["exit_time"])
            hours = (
                round((exit_dt - entry_dt).total_seconds() / 3600, 2)
                if entry_dt and exit_dt
                else None
            )
            records.append({
                "Name": row["name"],
                "Department": row["department"],
                "Status": "Late" if row["was_late"] else "Present",
                "Entry Time": row["entry_time"] or "",
                "Exit Time": row["exit_time"] or "",
                "Hours Worked": hours or "",
                "Late": bool(row["was_late"]),
            })

        for user in all_users:
            if user["name"] not in present_names:
                records.append({
                    "Name": user["name"],
                    "Department": user["department"],
                    "Status": "Absent",
                    "Entry Time": "",
                    "Exit Time": "",
                    "Hours Worked": "",
                    "Late": False,
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Weekly / monthly attendance percentage                               #
    # ------------------------------------------------------------------ #

    def attendance_percentage(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: Name, Department, Days Present,
        Total Days, Attendance %, Avg Hours.
        """
        if end_date is None:
            end_date = date.today().isoformat()
        if start_date is None:
            start_date = (date.today() - timedelta(days=30)).isoformat()

        # Count working days in range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        total_days = sum(
            1 for d in _date_range(start_dt, end_dt)
            if d.weekday() < 5  # Mon–Fri
        )
        if total_days == 0:
            total_days = 1

        rows = self._db.fetchall(
            """
            SELECT
                u.name, u.department,
                COUNT(DISTINCT a.date) AS days_present,
                AVG(
                    CASE WHEN a.event_type='EXIT' THEN
                        (CAST(strftime('%H', a.entry_time) AS REAL) * 60 +
                         CAST(strftime('%M', a.entry_time) AS REAL)) -
                        (SELECT CAST(strftime('%H', a2.entry_time) AS REAL)*60 +
                                CAST(strftime('%M', a2.entry_time) AS REAL)
                         FROM attendance a2
                         WHERE a2.user_id = a.user_id AND a2.date = a.date
                           AND a2.event_type = 'ENTRY'
                         ORDER BY a2.entry_time
                         LIMIT 1)
                    END
                ) / 60.0 AS avg_hours
            FROM attendance a
            JOIN users u ON u.id = a.user_id
            WHERE a.date BETWEEN ? AND ?
            GROUP BY a.user_id
            ORDER BY u.name
            """,
            (start_date, end_date),
        )

        present_ids = {r["name"] for r in rows}
        all_users = self._db.fetchall(
            "SELECT name, department FROM users WHERE active = 1"
        )

        records = [
            {
                "Name": r["name"],
                "Department": r["department"],
                "Days Present": r["days_present"],
                "Total Days": total_days,
                "Attendance %": round(r["days_present"] / total_days * 100, 1),
                "Avg Hours/Day": round(r["avg_hours"] or 0, 2),
            }
            for r in rows
        ]

        for u in all_users:
            if u["name"] not in present_ids:
                records.append({
                    "Name": u["name"],
                    "Department": u["department"],
                    "Days Present": 0,
                    "Total Days": total_days,
                    "Attendance %": 0.0,
                    "Avg Hours/Day": 0.0,
                })

        return pd.DataFrame(records).sort_values("Attendance %", ascending=False)

    # ------------------------------------------------------------------ #
    # Poor attendance detection                                            #
    # ------------------------------------------------------------------ #

    def poor_attendance_report(self, threshold_pct: float = 75.0) -> pd.DataFrame:
        """
        Return employees with attendance below *threshold_pct* over last 30 days.
        """
        df = self.attendance_percentage()
        return df[df["Attendance %"] < threshold_pct].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Late arrivals                                                        #
    # ------------------------------------------------------------------ #

    def late_arrivals_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = date.today().isoformat()
        if start_date is None:
            start_date = (date.today() - timedelta(days=30)).isoformat()

        rows = self._db.fetchall(
            """
            SELECT u.name, u.department, a.date, a.entry_time
            FROM attendance a
            JOIN users u ON u.id = a.user_id
            WHERE a.date BETWEEN ? AND ? AND a.status = 'LATE'
            ORDER BY a.date DESC, a.entry_time
            """,
            (start_date, end_date),
        )
        return pd.DataFrame([dict(r) for r in rows])

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def export_daily_csv(self, report_date: Optional[str] = None) -> str:
        """Export daily report to CSV; return file path."""
        df = self.daily_report(report_date)
        if report_date is None:
            report_date = date.today().isoformat()
        path = str(
            Path(self._cfg.exports_dir) / f"attendance_{report_date}.csv"
        )
        df.to_csv(path, index=False)
        logger.info("Exported daily CSV to {}", path)
        return path

    def export_summary_excel(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """Export multi-sheet Excel with daily + percentage + late reports."""
        if end_date is None:
            end_date = date.today().isoformat()
        if start_date is None:
            start_date = (date.today() - timedelta(days=30)).isoformat()

        path = str(
            Path(self._cfg.exports_dir)
            / f"report_{start_date}_to_{end_date}.xlsx"
        )
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.daily_report().to_excel(writer, sheet_name="Today", index=False)
            self.attendance_percentage(start_date, end_date).to_excel(
                writer, sheet_name="Summary", index=False
            )
            self.late_arrivals_report(start_date, end_date).to_excel(
                writer, sheet_name="Late Arrivals", index=False
            )
            self.poor_attendance_report().to_excel(
                writer, sheet_name="Poor Attendance", index=False
            )

        logger.info("Exported Excel report to {}", path)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

#!/usr/bin/env bash
# Start AttendX
[ -d "venv" ] && source venv/bin/activate
mkdir -p data uploads logs
python main_web.py

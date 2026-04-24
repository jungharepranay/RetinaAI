"""
database.py
-----------
SQLite database layer for RetinAI dashboard system.

Uses aiosqlite for async operations within FastAPI.
Stores users, scan results, and clinician notes.

Tables:
    users           — registered users (patient / clinician)
    scans           — prediction results linked to users
    clinician_notes — clinician annotations on cases
"""

import os
import json
import aiosqlite
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "retinai.db")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")


async def init_db():
    """Create database tables if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('patient', 'clinician')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_name TEXT DEFAULT '',
                role TEXT NOT NULL,
                image_path TEXT,
                prediction_output TEXT,
                clinical_context TEXT,
                questionnaire_data TEXT DEFAULT '{}',
                risk_priority TEXT DEFAULT 'routine',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Migration: add questionnaire_data column if upgrading from older schema
        try:
            await db.execute(
                "ALTER TABLE scans ADD COLUMN questionnaire_data TEXT DEFAULT '{}'"
            )
        except Exception:
            pass  # Column already exists

        await db.execute("""
            CREATE TABLE IF NOT EXISTS clinician_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER NOT NULL,
                clinician_id INTEGER NOT NULL,
                note_type TEXT DEFAULT 'general',
                note_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (scan_id) REFERENCES scans(id),
                FOREIGN KEY (clinician_id) REFERENCES users(id)
            )
        """)

        await db.commit()
        print("[database] Tables initialized")


# ================================================================== #
#  USER OPERATIONS                                                     #
# ================================================================== #

async def create_user(name: str, email: str, password_hash: str, role: str) -> dict:
    """Create a new user. Returns the user dict or raises on duplicate email."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, role),
        )
        await db.commit()
        user_id = cursor.lastrowid

    return {"id": user_id, "name": name, "email": email, "role": role}


async def get_user_by_email(email: str) -> dict | None:
    """Look up a user by email. Returns dict or None."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, name, email, password_hash, role, created_at FROM users WHERE email = ?",
            (email,),
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)
    return None


async def get_user_by_id(user_id: int) -> dict | None:
    """Look up a user by ID. Returns dict or None."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, name, email, role, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)
    return None


# ================================================================== #
#  SCAN OPERATIONS                                                     #
# ================================================================== #

async def create_scan(
    user_id: int,
    role: str,
    image_path: str = None,
    prediction_output: dict = None,
    clinical_context: dict = None,
    risk_priority: str = "routine",
    patient_name: str = "",
) -> int:
    """Save a scan result. Returns the new scan ID."""
    pred_json = json.dumps(prediction_output) if prediction_output else "{}"
    ctx_json = json.dumps(clinical_context) if clinical_context else "{}"

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO scans
               (user_id, patient_name, role, image_path, prediction_output,
                clinical_context, risk_priority)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, patient_name, role, image_path, pred_json,
             ctx_json, risk_priority),
        )
        await db.commit()
        return cursor.lastrowid


async def update_scan_questionnaire(scan_id: int, questionnaire_data: dict) -> None:
    """Update a scan with post-prediction questionnaire data."""
    q_json = json.dumps(questionnaire_data) if questionnaire_data else "{}"
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE scans SET questionnaire_data = ? WHERE id = ?",
            (q_json, scan_id),
        )
        await db.commit()


async def get_scans_for_user(user_id: int, limit: int = 10) -> list:
    """Get scans belonging to a specific user, newest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT id, user_id, patient_name, role, image_path,
                      prediction_output, clinical_context, risk_priority, created_at
               FROM scans WHERE user_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [_parse_scan_row(dict(r)) for r in rows]


async def get_all_scans(sort_by: str = "latest", limit: int = 50) -> list:
    """Get all scans (clinician view). Supports sorting by 'latest' or 'risk'."""
    order = "s.created_at DESC"
    if sort_by == "risk":
        order = """CASE s.risk_priority
                       WHEN 'urgent' THEN 1
                       WHEN 'semi-urgent' THEN 2
                       ELSE 3
                   END, s.created_at DESC"""

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            f"""SELECT s.id, s.user_id, s.patient_name, s.role, s.image_path,
                       s.prediction_output, s.clinical_context, s.risk_priority,
                       s.created_at, u.name as uploader_name, u.email as uploader_email
                FROM scans s
                LEFT JOIN users u ON s.user_id = u.id
                ORDER BY {order} LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [_parse_scan_row(dict(r)) for r in rows]


async def get_scan_by_id(scan_id: int) -> dict | None:
    """Get a single scan by ID with user info."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT s.*, u.name as uploader_name, u.email as uploader_email
               FROM scans s
               LEFT JOIN users u ON s.user_id = u.id
               WHERE s.id = ?""",
            (scan_id,),
        )
        row = await cursor.fetchone()
        if row:
            return _parse_scan_row(dict(row))
    return None


def _parse_scan_row(row: dict) -> dict:
    """Parse JSON fields in a scan row."""
    try:
        row["prediction_output"] = json.loads(row.get("prediction_output") or "{}")
    except (json.JSONDecodeError, TypeError):
        row["prediction_output"] = {}
    try:
        row["clinical_context"] = json.loads(row.get("clinical_context") or "{}")
    except (json.JSONDecodeError, TypeError):
        row["clinical_context"] = {}
    return row


# ================================================================== #
#  CLINICIAN NOTES                                                     #
# ================================================================== #

async def add_clinician_note(
    scan_id: int,
    clinician_id: int,
    note_type: str = "general",
    note_text: str = "",
) -> int:
    """Add a clinician note to a scan."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO clinician_notes (scan_id, clinician_id, note_type, note_text)
               VALUES (?, ?, ?, ?)""",
            (scan_id, clinician_id, note_type, note_text),
        )
        await db.commit()
        return cursor.lastrowid


async def get_notes_for_scan(scan_id: int) -> list:
    """Get all clinician notes for a scan."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT cn.*, u.name as clinician_name
               FROM clinician_notes cn
               LEFT JOIN users u ON cn.clinician_id = u.id
               WHERE cn.scan_id = ?
               ORDER BY cn.created_at DESC""",
            (scan_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


# ================================================================== #
#  DASHBOARD STATISTICS                                                #
# ================================================================== #

async def get_dashboard_stats(user_id: int = None) -> dict:
    """Get dashboard statistics. If user_id given, scoped to that user."""
    async with aiosqlite.connect(DB_PATH) as db:
        if user_id:
            # Patient stats
            cursor = await db.execute(
                "SELECT COUNT(*) FROM scans WHERE user_id = ?", (user_id,)
            )
            total_scans = (await cursor.fetchone())[0]

            cursor = await db.execute(
                """SELECT risk_priority, prediction_output FROM scans
                   WHERE user_id = ? ORDER BY created_at DESC LIMIT 1""",
                (user_id,),
            )
            last_row = await cursor.fetchone()
            last_risk = last_row[0] if last_row else None
            last_prediction = None
            if last_row and last_row[1]:
                try:
                    last_prediction = json.loads(last_row[1])
                except (json.JSONDecodeError, TypeError):
                    pass

            return {
                "total_scans": total_scans,
                "last_risk": last_risk,
                "last_prediction": last_prediction,
            }
        else:
            # Clinician stats
            cursor = await db.execute("SELECT COUNT(DISTINCT user_id) FROM scans")
            total_patients = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM scans")
            total_cases = (await cursor.fetchone())[0]

            cursor = await db.execute(
                "SELECT COUNT(*) FROM scans WHERE risk_priority = 'urgent'"
            )
            high_risk = (await cursor.fetchone())[0]

            return {
                "total_patients": total_patients,
                "total_cases": total_cases,
                "high_risk_cases": high_risk,
            }

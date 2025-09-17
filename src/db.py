import sqlite3
from typing import Dict, Any, List
import os
import pandas as pd


DB_PATH = os.environ.get('KMRL_DB_PATH', 'kmrl.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS trains (
            train_id TEXT PRIMARY KEY,
            fitness_valid_until TEXT,
            mileage_km REAL,
            branding_hours_left REAL,
            cleaning_slot_id TEXT,
            bay_geometry_score REAL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS job_cards (
            train_id TEXT PRIMARY KEY,
            job_card_status TEXT,
            last_updated TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS cleaning_slots (
            slot_id TEXT PRIMARY KEY,
            available_bays INTEGER,
            priority TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS bay_config (
            bay_id TEXT PRIMARY KEY,
            bay_type TEXT,
            max_capacity INTEGER,
            geometry_score REAL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS branding_contracts (
            contract_id TEXT PRIMARY KEY,
            brand TEXT,
            train_id TEXT,
            hours_committed REAL,
            start_date TEXT,
            end_date TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS branding_exposure (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            train_id TEXT,
            brand TEXT,
            date TEXT,
            hours REAL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            train_id TEXT,
            inducted INTEGER,
            failures INTEGER,
            notes TEXT
        )
    ''')
    # Model registry for trained models
    cur.execute('''
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id TEXT PRIMARY KEY,
            model_name TEXT,
            version TEXT,
            created_at TEXT,
            created_by TEXT,
            artifact_path TEXT,
            params_json TEXT,
            is_active INTEGER DEFAULT 0
        )
    ''')
    # Model metrics history
    cur.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT,
            timestamp TEXT,
            metric_name TEXT,
            metric_value REAL
        )
    ''')
    # Final plan lock audit
    cur.execute('''
        CREATE TABLE IF NOT EXISTS plan_lock_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT,
            locked_by TEXT,
            locked_at TEXT,
            depot_id TEXT,
            snapshot_json TEXT
        )
    ''')
    # Approvals master table (current state)
    cur.execute('''
        CREATE TABLE IF NOT EXISTS approvals (
            plan_id TEXT PRIMARY KEY,
            submitted_by TEXT,
            submitted_at TEXT,
            status TEXT,
            decided_by TEXT,
            decided_at TEXT,
            reason TEXT,
            plan_json TEXT
        )
    ''')
    # Approvals audit history (append-only)
    cur.execute('''
        CREATE TABLE IF NOT EXISTS approvals_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT,
            event TEXT,
            actor TEXT,
            event_time TEXT,
            details TEXT
        )
    ''')
    # Integration heartbeat table for external data freshness
    cur.execute('''
        CREATE TABLE IF NOT EXISTS integration_heartbeat (
            source TEXT PRIMARY KEY,
            last_heartbeat TEXT,
            status TEXT,
            detail TEXT
        )
    ''')
    conn.commit()
    conn.close()


def upsert(table: str, rows: List[Dict[str, Any]], key_cols: List[str]):
    if not rows:
        return
    conn = get_connection()
    cur = conn.cursor()
    cols = list(rows[0].keys())
    placeholders = ','.join(['?'] * len(cols))
    update_clause = ','.join([f"{c}=excluded.{c}" for c in cols if c not in key_cols])
    sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) " \
          f"ON CONFLICT({','.join(key_cols)}) DO UPDATE SET {update_clause}"
    values = [tuple(r.get(c) for c in cols) for r in rows]
    cur.executemany(sql, values)
    conn.commit()
    conn.close()


def fetch_df(table: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def fetch_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def insert_rows(table: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    conn = get_connection()
    cur = conn.cursor()
    cols = list(rows[0].keys())
    placeholders = ','.join(['?'] * len(cols))
    sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    values = [tuple(r.get(c) for c in cols) for r in rows]
    cur.executemany(sql, values)
    conn.commit()
    conn.close()


def upsert_heartbeat(source: str, status: str, detail: str = ""):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO integration_heartbeat (source, last_heartbeat, status, detail)
        VALUES (?, datetime('now'), ?, ?)
        ON CONFLICT(source) DO UPDATE SET
            last_heartbeat = excluded.last_heartbeat,
            status = excluded.status,
            detail = excluded.detail
    ''', (source, status, detail))
    conn.commit()
    conn.close()


def get_heartbeats() -> pd.DataFrame:
    return fetch_query("SELECT * FROM integration_heartbeat", ())


def bootstrap_from_csv_if_empty(csv_paths: Dict[str, str]):
    """Load CSVs into DB tables if tables are empty."""
    init_db()
    # trains
    if fetch_df('trains').empty and os.path.exists(csv_paths.get('trains', '')):
        df = pd.read_csv(csv_paths['trains'])
        upsert('trains', df.to_dict(orient='records'), ['train_id'])
    # job_cards
    if fetch_df('job_cards').empty and os.path.exists(csv_paths.get('job_cards', '')):
        df = pd.read_csv(csv_paths['job_cards'])
        df['last_updated'] = pd.Timestamp.utcnow().isoformat()
        upsert('job_cards', df.to_dict(orient='records'), ['train_id'])
    # cleaning_slots
    if fetch_df('cleaning_slots').empty and os.path.exists(csv_paths.get('cleaning_slots', '')):
        df = pd.read_csv(csv_paths['cleaning_slots'])
        upsert('cleaning_slots', df.to_dict(orient='records'), ['slot_id'])
    # bay_config
    if fetch_df('bay_config').empty and os.path.exists(csv_paths.get('bay_config', '')):
        df = pd.read_csv(csv_paths['bay_config'])
        upsert('bay_config', df.to_dict(orient='records'), ['bay_id'])



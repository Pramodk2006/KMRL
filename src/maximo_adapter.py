import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
from .db import upsert, upsert_heartbeat


class MaximoAdapter:
    """Simple Maximo adapter that reads a local JSON/CSV export and upserts job-cards.
    Replace read_export() with true API calls later.
    """

    def __init__(self, export_path: str = None):
        # Allow configuration via env var
        self.export_path = export_path or os.environ.get('KMRL_MAXIMO_EXPORT', '')

    def read_export(self) -> pd.DataFrame:
        path = self.export_path
        if not path or not os.path.exists(path):
            return pd.DataFrame(columns=['train_id', 'job_card_status'])
        if path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        # assume CSV
        return pd.read_csv(path)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Ensure required columns and types
        out = pd.DataFrame()
        out['train_id'] = df['train_id'].astype(str)
        # Map status fields if needed
        if 'status' in df.columns and 'job_card_status' not in df.columns:
            out['job_card_status'] = df['status'].astype(str).str.lower().map({'open': 'open', 'closed': 'closed'}).fillna('closed')
        else:
            out['job_card_status'] = df['job_card_status'].astype(str).str.lower().fillna('closed')
        out['last_updated'] = datetime.utcnow().isoformat()
        return out

    def upsert_job_cards(self, rows: List[Dict]):
        if not rows:
            return 0
        upsert('job_cards', rows, ['train_id'])
        return len(rows)

    def refresh(self) -> int:
        df = self.read_export()
        df = self.normalize(df)
        count = self.upsert_job_cards(df.to_dict(orient='records'))
        # Write heartbeat to DB
        try:
            status = 'ok' if count >= 0 else 'error'
            upsert_heartbeat('maximo', status, f"rows={count}")
        except Exception:
            pass
        return count



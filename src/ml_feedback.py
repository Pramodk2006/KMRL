from datetime import datetime, date
from typing import Dict, Any
import pandas as pd
import numpy as np
from .db import fetch_df, upsert, init_db


def record_nightly_outcomes(inducted_trains: pd.DataFrame, failures_df: pd.DataFrame = None, notes: str = "") -> int:
    """Persist nightly outcomes to DB for ML feedback.

    inducted_trains: DataFrame or list of dicts with train_id and inducted=1
    failures_df: optional DataFrame with train_id, failure_flag (1/0)
    """
    init_db()
    if isinstance(inducted_trains, list):
        df = pd.DataFrame(inducted_trains)
    else:
        df = inducted_trains.copy()
    df = df[['train_id']].drop_duplicates()
    df['date'] = date.today().isoformat()
    df['inducted'] = 1
    df['failures'] = 0
    if failures_df is not None and not failures_df.empty:
        f = failures_df[['train_id', 'failure_flag']].groupby('train_id', as_index=False)['failure_flag'].max()
        df = df.merge(f, on='train_id', how='left')
        df['failures'] = df['failure_flag'].fillna(0).astype(int)
        df.drop(columns=['failure_flag'], inplace=True)
    rows = df[['date', 'train_id', 'inducted', 'failures']].to_dict(orient='records')
    # upsert using (date, train_id) uniqueness emulated via update on conflict; outcomes table uses autoincrement, so we insert many
    upsert('outcomes', [{'date': r['date'], 'train_id': r['train_id'], 'inducted': r['inducted'], 'failures': r['failures'], 'notes': notes} for r in rows], ['id'])
    return len(rows)


def compute_drift_metrics() -> Dict[str, Any]:
    """Compute simple model/data drift indicators from stored outcomes vs. recent predictions.
    This is a scaffold; integrate with full prediction logs later.
    """
    outcomes = fetch_df('outcomes')
    if outcomes.empty:
        return {'status': 'no_data'}
    # Basic aggregates
    outcomes['date'] = pd.to_datetime(outcomes['date'])
    last_14 = outcomes[outcomes['date'] >= (pd.Timestamp.today() - pd.Timedelta(days=14))]
    fail_rate_14 = last_14['failures'].mean() if not last_14.empty else 0.0
    fail_rate_all = outcomes['failures'].mean()
    drift = float(fail_rate_14 - fail_rate_all)
    return {
        'window_days': 14,
        'failure_rate_14d': float(fail_rate_14),
        'failure_rate_all': float(fail_rate_all),
        'drift_delta': drift,
        'status': 'ok'
    }


def retrain_predictive_model_if_needed(threshold: float = 0.05) -> Dict[str, Any]:
    """Placeholder retraining trigger based on drift.
    In real use, call your training pipeline and update the model registry.
    """
    metrics = compute_drift_metrics()
    if metrics.get('status') != 'ok':
        return {'status': 'skipped', 'reason': 'no_data'}
    if abs(metrics['drift_delta']) >= threshold:
        # Here you would train and persist the new model
        return {'status': 'retrain_triggered', 'metrics': metrics, 'threshold': threshold}
    return {'status': 'stable', 'metrics': metrics, 'threshold': threshold}



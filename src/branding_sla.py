from datetime import datetime, date
from typing import List, Dict, Any
import os
import pandas as pd
from .db import upsert, fetch_df, init_db


def ingest_contracts_from_csv(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    init_db()
    df = pd.read_csv(csv_path)
    # Expected columns: contract_id, brand, train_id, hours_committed, start_date, end_date
    required = ['contract_id', 'brand', 'train_id', 'hours_committed', 'start_date', 'end_date']
    for col in required:
        if col not in df.columns:
            df[col] = None
    rows = df[required].to_dict(orient='records')
    upsert('branding_contracts', rows, ['contract_id'])
    return len(rows)


def ingest_contracts_from_json(contracts: List[Dict[str, Any]]) -> int:
    if not contracts:
        return 0
    init_db()
    # Normalize
    rows = []
    for c in contracts:
        rows.append({
            'contract_id': c.get('contract_id'),
            'brand': c.get('brand'),
            'train_id': c.get('train_id'),
            'hours_committed': c.get('hours_committed', 0),
            'start_date': c.get('start_date'),
            'end_date': c.get('end_date')
        })
    upsert('branding_contracts', rows, ['contract_id'])
    return len(rows)


def record_exposure_for_inductions(inducted_trains: List[Dict[str, Any]], hours: float = 8.0, exposure_date: str = None) -> int:
    """Record nightly exposure hours for inducted trains that have branding contracts."""
    if not inducted_trains:
        return 0
    init_db()
    contracts = fetch_df('branding_contracts')
    if contracts.empty:
        return 0
    exposure_date = exposure_date or date.today().isoformat()
    rows = []
    for t in inducted_trains:
        train_id = t.get('train_id')
        if not train_id:
            continue
        # Find contracts for this train active on date
        act = contracts[contracts['train_id'] == train_id]
        if act.empty:
            continue
        for _, c in act.iterrows():
            # Optional date filtering if dates provided
            try:
                if pd.notna(c['start_date']) and pd.to_datetime(exposure_date) < pd.to_datetime(c['start_date']):
                    continue
                if pd.notna(c['end_date']) and pd.to_datetime(exposure_date) > pd.to_datetime(c['end_date']):
                    continue
            except Exception:
                pass
            rows.append({
                'train_id': train_id,
                'brand': c.get('brand'),
                'date': exposure_date,
                'hours': hours
            })
    if not rows:
        return 0
    # No ON CONFLICT for autoincrement, use simple inserts via upsert on composite (train_id, brand, date)
    # Create a temp key by combining fields
    # Since upsert() expects a PK/unique key, we will emulate by aggregating before insert to avoid collisions.
    df = pd.DataFrame(rows)
    agg = df.groupby(['train_id', 'brand', 'date'], as_index=False)['hours'].sum()
    # Create a composite key emulation by inserting rows; table doesn't have unique constraint so duplicates won't break
    upsert('branding_exposure', agg.to_dict(orient='records'), ['id'])  # 'id' autoincrement; upsert behaves like insert
    return len(agg)


def get_sla_status(nightly_hours_budget: float = 8.0) -> Dict[str, Any]:
    """Compute SLA status per contract and overall alerts."""
    contracts = fetch_df('branding_contracts')
    exposure = fetch_df('branding_exposure')
    if contracts.empty:
        return {'contracts': [], 'alerts': [], 'summary': {'total_contracts': 0}}
    if exposure.empty:
        exposure = pd.DataFrame(columns=['train_id', 'brand', 'date', 'hours'])
    # Sum exposure by (train_id, brand)
    delivered = exposure.groupby(['train_id', 'brand'], as_index=False)['hours'].sum().rename(columns={'hours': 'hours_delivered'})
    merged = contracts.merge(delivered, on=['train_id', 'brand'], how='left')
    merged['hours_delivered'] = merged['hours_delivered'].fillna(0.0)
    merged['hours_committed'] = merged['hours_committed'].fillna(0.0)
    # Remaining and projections
    today = pd.to_datetime(date.today())
    def days_left(row):
        try:
            if pd.isna(row['end_date']):
                return 0
            end = pd.to_datetime(row['end_date'])
            d = (end - today).days
            return max(d, 0)
        except Exception:
            return 0
    merged['days_left'] = merged.apply(days_left, axis=1)
    merged['remaining_hours'] = (merged['hours_committed'] - merged['hours_delivered']).clip(lower=0)
    merged['max_possible'] = merged['days_left'] * nightly_hours_budget
    merged['projected_breach'] = merged['remaining_hours'] > merged['max_possible']
    merged['pct_delivered'] = (merged['hours_delivered'] / merged['hours_committed']).replace([pd.NaT, pd.NA], 0).fillna(0)
    # Build output
    contracts_out = []
    alerts = []
    for _, r in merged.iterrows():
        item = {
            'contract_id': r.get('contract_id'),
            'brand': r.get('brand'),
            'train_id': r.get('train_id'),
            'hours_committed': float(r.get('hours_committed', 0)),
            'hours_delivered': float(r.get('hours_delivered', 0)),
            'remaining_hours': float(r.get('remaining_hours', 0)),
            'days_left': int(r.get('days_left', 0)),
            'projected_breach': bool(r.get('projected_breach', False)),
            'pct_delivered': float(r.get('pct_delivered', 0))
        }
        contracts_out.append(item)
        if item['projected_breach']:
            alerts.append({
                'type': 'branding_sla_breach',
                'contract_id': item['contract_id'],
                'message': f"Projected SLA breach for {item['brand']} on {item['train_id']} (remaining {item['remaining_hours']:.1f}h, days_left {item['days_left']})"
            })
    summary = {
        'total_contracts': len(contracts_out),
        'breach_count': len(alerts),
        'total_remaining_hours': float(merged['remaining_hours'].sum())
    }
    return {'contracts': contracts_out, 'alerts': alerts, 'summary': summary}



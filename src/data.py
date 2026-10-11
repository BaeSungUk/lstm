"""Download, audit, chronological session split, and next-click examples."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

URL = 'https://archive.ics.uci.edu/static/public/553/clickstream+data+for+online+shopping.zip'
CATEGORY = 'page 1 (main category)'
SESSION = 'session ID'
NAMES = ['Trousers', 'Skirts', 'Blouses', 'Sale']


def download_data(root: Path) -> Path:
    """Cache the original CSV; extract only known file types, never ZIP paths."""
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'e-shop clothing 2008.csv'
    if target.exists():
        return target
    try:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            filename = next(n for n in archive.namelist() if n.endswith('.csv'))
            content = archive.read(filename)
        # Atomic replacement avoids treating interrupted downloads as valid data.
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(target)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, StopIteration) as exc:
        raise RuntimeError(
            f'UCI download failed: {exc}. Check your connection and retry, or place '
            f'the original CSV at {target}. Source: {URL}'
        ) from exc
    return target


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Exclude a whole session when essential click information is ambiguous."""
    required = ['year', 'month', 'day', SESSION, 'order', CATEGORY]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    audit = {'raw_rows': len(frame), 'raw_missing_cells': int(frame.isna().sum().sum())}
    df = frame.drop_duplicates().copy()
    audit['duplicate_rows'] = len(frame) - len(df)
    for column in required:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    valid_id = df[SESSION].notna() & np.isfinite(df[SESSION]) & (df[SESSION] % 1 == 0)
    audit['invalid_session_id_rows'] = int((~valid_id).sum())
    df = df.loc[valid_id].copy()
    df[SESSION] = df[SESSION].astype('int64')
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    bad = (df['date'].isna() | ~df[CATEGORY].isin([1, 2, 3, 4])
           | df['order'].isna() | (df['order'] < 1) | (df['order'] % 1 != 0)
           | ~np.isfinite(df['order']) | df.duplicated([SESSION, 'order'], keep=False))
    bad_ids = set(df.loc[bad, SESSION])
    audit['invalid_sessions'] = len(bad_ids)
    audit['invalid_session_rows'] = int(df[SESSION].isin(bad_ids).sum())
    df = df.loc[~df[SESSION].isin(bad_ids)].copy()
    df['order'] = df['order'].astype('int64')
    df[CATEGORY] = df[CATEGORY].astype('int64')
    df = df.sort_values([SESSION, 'order']).reset_index(drop=True)
    # A date moving backwards within a click sequence makes temporal splitting unsafe.
    backwards = df.groupby(SESSION)['date'].diff().dt.total_seconds().lt(0)
    backwards_ids = set(df.loc[backwards, SESSION])
    audit['backwards_date_sessions'] = len(backwards_ids)
    audit['backwards_date_rows'] = int(df[SESSION].isin(backwards_ids).sum())
    df = df.loc[~df[SESSION].isin(backwards_ids)].reset_index(drop=True)
    audit.update(clean_rows=len(df), clean_sessions=int(df[SESSION].nunique()))
    return df, audit


def split_sessions(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """About 70/15/15% of sessions; never split a date or a session across sets."""
    spans = frame.groupby(SESSION)['date'].agg(['min', 'max'])
    counts = spans.groupby('min').size().sort_index()
    if len(counts) < 3:
        raise ValueError('At least three session-start dates are required for splitting.')
    cumulative = counts.cumsum().to_numpy()
    first = min(int(np.searchsorted(cumulative, len(spans) * .70)), len(counts) - 3)
    second = min(max(int(np.searchsorted(cumulative, len(spans) * .85)), first + 1), len(counts) - 2)
    cut1, cut2 = counts.index[first], counts.index[second]
    crossing = ((spans['min'].le(cut1) & spans['max'].gt(cut1))
                | (spans['min'].le(cut2) & spans['max'].gt(cut2)))
    eligible = spans.loc[~crossing]
    ids = {
        'train': eligible.index[eligible['min'].le(cut1)],
        'validation': eligible.index[eligible['min'].gt(cut1) & eligible['min'].le(cut2)],
        'test': eligible.index[eligible['min'].gt(cut2)],
    }
    parts = {name: frame.loc[frame[SESSION].isin(group)].copy() for name, group in ids.items()}
    audit = {'train_end': str(cut1.date()), 'validation_end': str(cut2.date()),
             'cross_boundary_sessions': int(crossing.sum()),
             'cross_boundary_rows': int(frame[SESSION].isin(spans.index[crossing]).sum())}
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f'Empty {name} split; more dates are required.')
        audit[name] = {'rows': len(part), 'sessions': int(part[SESSION].nunique()),
                       'start': str(part.date.min().date()), 'end': str(part.date.max().date())}
    return parts, audit


@dataclass
class Examples:
    x: np.ndarray
    lengths: np.ndarray
    y: np.ndarray
    last: np.ndarray
    session_ids: np.ndarray
    orders: np.ndarray


def make_examples(frame: pd.DataFrame, length: int) -> Examples:
    """For each click after the first, use only earlier clicks from its session."""
    if length < 1:
        raise ValueError('History length must be positive.')
    windows, lengths, targets, last, sessions, orders = [], [], [], [], [], []
    eye = np.eye(4, dtype=np.float32)
    for session, group in frame.groupby(SESSION, sort=True):
        group = group.sort_values('order')
        categories = group[CATEGORY].to_numpy(dtype=np.int64) - 1
        click_orders = group['order'].to_numpy(dtype=np.int64)
        for t in range(1, len(categories)):
            history = categories[max(0, t - length):t]
            window = np.zeros((length, 4), dtype=np.float32)
            window[:len(history)] = eye[history]
            windows.append(window)
            lengths.append(len(history))
            targets.append(categories[t])
            last.append(categories[t - 1])
            sessions.append(session)
            orders.append(click_orders[t])
    if not windows:
        raise ValueError('No next-click examples: need sessions with at least two clicks.')
    return Examples(np.stack(windows), *[np.asarray(a, dtype=np.int64)
                    for a in [lengths, targets, last, sessions, orders]])


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

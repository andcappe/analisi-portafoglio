"""
Dashboard Analisi Portafoglio — versione client standalone
Replica esatta della Tab 1 di ir_fe_14.py
"""

import io
import json
import base64
import time
import threading
import os
import uuid
import requests
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, html, dcc, Input, Output, State, ALL, callback_context, no_update
from dash.exceptions import PreventUpdate
from flask import send_file as flask_send_file

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
_EXTERNAL_STYLESHEETS = [
    'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
]

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=_EXTERNAL_STYLESHEETS)
app.server.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Analisi di Portafoglio</title>
{%favicon%}
{%css%}
<style>
  [data-tooltip] { position: relative; }
  [data-tooltip]::after {
    content: attr(data-tooltip);
    position: absolute;
    left: 100%;
    top: 50%;
    transform: translateY(-50%);
    background: #1a3a5c;
    color: #fff;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    white-space: nowrap;
    z-index: 9999;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    margin-left: 6px;
  }
  [data-tooltip]:hover::after { opacity: 1; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
'''

# Percorsi file
_XLSX        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TARBIUTH.xlsx')
_PROFILO_HTML = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '..', 'profilo', 'index.html'))

# Rotta Flask per servire la pagina profilo
@app.server.route('/sito')
def serve_profilo():
    return flask_send_file(_PROFILO_HTML)

@app.server.route('/test-yf')
def test_yf():
    import json as _json
    done = threading.Event()
    result = [None]
    def _go():
        try:
            df = yf.download('AAPL', start='2024-01-01', auto_adjust=True,
                             progress=False, threads=False)
            result[0] = {'ok': True, 'rows': len(df), 'cols': list(df.columns)}
        except Exception as e:
            result[0] = {'ok': False, 'error': str(e)}
        finally:
            done.set()
    threading.Thread(target=_go, daemon=True).start()
    done.wait(timeout=30)
    return _json.dumps(result[0] or {'ok': False, 'error': 'timeout'})

# ─────────────────────────────────────────────────────────────────────────────
# Colori
# ─────────────────────────────────────────────────────────────────────────────
color_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#a6cee3',
    '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99',
    '#191970', '#006400', '#8B0000', '#4B0082', '#2F4F4F',
    '#8B4513', '#FF1493', '#696969', '#556B2F', '#008B8B',
    '#E9967A', '#90EE90', '#B0C4DE', '#D3D3D3'
]

# ─────────────────────────────────────────────────────────────────────────────
# Calcoli
# ─────────────────────────────────────────────────────────────────────────────
def calculate_rolling_information_ratio(asset_returns, benchmark_returns, window):
    active_return = asset_returns - benchmark_returns.to_numpy()
    rolling_mean  = active_return.rolling(window=window).mean()
    rolling_std   = active_return.rolling(window=window).std()
    return (rolling_mean / rolling_std) * np.sqrt(252)

def calculate_rolling_sharpe_ratio(returns, window):
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std  = returns.rolling(window=window).std()
    return (rolling_mean / rolling_std) * np.sqrt(252)

def calculate_tracking_error_volatility(asset_returns, benchmark_returns, window):
    active_return = asset_returns - benchmark_returns
    return active_return.rolling(window=window).std() * np.sqrt(252)

def calculate_drawdown(returns_series):
    cumulative   = (1 + returns_series).cumprod()
    rolling_max  = cumulative.cummax()
    return (cumulative - rolling_max) / rolling_max

def calculate_rolling_cvar(returns_series, window, tail_pct=0.05):
    def _cvar(w):
        n_tail = max(1, int(np.floor(len(w) * tail_pct)))
        return np.sort(w)[:n_tail].mean()
    min_p = max(10, window // 2)
    return returns_series.rolling(window, min_periods=min_p).apply(_cvar, raw=True)

def _rolling_volatility(returns_series, window):
    return returns_series.rolling(window, min_periods=window // 2).std() * np.sqrt(252)

# ─────────────────────────────────────────────────────────────────────────────
# Cache DataFrame
# ─────────────────────────────────────────────────────────────────────────────
_DF_CACHE: dict = {}

def _df_key(json_str):
    return json_str[:120]

def _get_df(json_str):
    if not json_str:
        return None
    key = _df_key(json_str)
    if key not in _DF_CACHE:
        df = pd.read_json(io.StringIO(json_str), orient='split')
        df.index = pd.to_datetime(df.index)
        _DF_CACHE[key] = df
        if len(_DF_CACHE) > 20:
            oldest = next(iter(_DF_CACHE))
            del _DF_CACHE[oldest]
    return _DF_CACHE[key].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Download worker (background thread)
# ─────────────────────────────────────────────────────────────────────────────
DOWNLOAD_BATCH_SIZE = 10   # ticker per batch → 18 chiamate invece di 177
DOWNLOAD_TIMEOUT    = 60   # secondi per batch (più ticker = più tempo)

_DL_STATE  = {'status': 'idle', 'current': 0, 'total': 0, 'errors': []}
_DL_BUFFER = {}
_DL_LOCK   = threading.Lock()


def _download_batch(batch_tickers, start_date):
    """Scarica un gruppo di ticker in una sola chiamata con timeout non bloccante."""
    result = [None]
    done   = threading.Event()

    def _fetch():
        try:
            result[0] = yf.download(
                batch_tickers, start=start_date,
                auto_adjust=True, progress=False, threads=False,
            )
        except Exception as e:
            print(f"⚠ Batch error: {e}")
        finally:
            done.set()

    threading.Thread(target=_fetch, daemon=True).start()

    if done.wait(timeout=DOWNLOAD_TIMEOUT):
        return result[0]
    print(f"⚠ Timeout batch: {batch_tickers[:3]}…")
    return None


def _extract_close(raw, ticker):
    """Estrae la serie Close da un DataFrame potenzialmente MultiIndex."""
    if raw is None or raw.empty:
        return None
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance 0.2.x: livello 0 = campo (Close/High/…), livello 1 = ticker
            if ('Close', ticker) in raw.columns:
                s = raw[('Close', ticker)]
            elif (ticker, 'Close') in raw.columns:
                s = raw[(ticker, 'Close')]
            else:
                return None
        else:
            s = raw['Close']
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.dropna().copy() if not s.dropna().empty else None
    except Exception as e:
        print(f"⚠ Estrazione Close {ticker}: {e}")
        return None


def _download_worker(tickers, descrizione, valuta, start_date='2016-01-01'):
    global _DL_STATE, _DL_BUFFER
    total = len(tickers)
    with _DL_LOCK:
        _DL_STATE  = {'status': 'running', 'current': 0, 'total': total, 'errors': []}
        _DL_BUFFER = {}

    try:
        # ── 1. FX (batch unico, non conta nel progresso) ──────────────────────
        fx_raw  = _download_batch(['EURUSD=X', 'EURGBP=X'], start_date)
        eurusd  = _extract_close(fx_raw, 'EURUSD=X')
        eurgbp  = _extract_close(fx_raw, 'EURGBP=X')

        # ── 2. Ticker principali in batch di DOWNLOAD_BATCH_SIZE ──────────────
        all_prices = {}
        for batch_start in range(0, total, DOWNLOAD_BATCH_SIZE):
            batch_t = tickers[batch_start: batch_start + DOWNLOAD_BATCH_SIZE]
            batch_d = descrizione[batch_start: batch_start + DOWNLOAD_BATCH_SIZE]
            batch_v = valuta[batch_start: batch_start + DOWNLOAD_BATCH_SIZE]

            raw = _download_batch(batch_t, start_date)

            for t, desc, curr in zip(batch_t, batch_d, batch_v):
                px = _extract_close(raw, t)
                if px is None:
                    with _DL_LOCK:
                        _DL_STATE['errors'].append(f"{desc}: nessun dato")
                    continue
                if curr == 'USD' and eurusd is not None:
                    px = px / eurusd.reindex(px.index).ffill()
                elif curr == 'GBP' and eurgbp is not None:
                    px = px / eurgbp.reindex(px.index).ffill()
                all_prices[desc] = px

            with _DL_LOCK:
                _DL_STATE['current'] = min(batch_start + len(batch_t), total)

            time.sleep(0.3)   # pausa cortese verso Yahoo

        # ── 3. Assembla DataFrame ─────────────────────────────────────────────
        if all_prices:
            original_prices = pd.DataFrame(all_prices)
            original_prices.index = pd.to_datetime(original_prices.index)
            original_prices = original_prices.ffill()
            close_returns = original_prices.pct_change(fill_method=None)
            with _DL_LOCK:
                _DL_BUFFER['original_prices'] = original_prices
                _DL_BUFFER['close_returns']   = close_returns
                _DL_STATE['status'] = 'done'
            print(f"✓ Download completato: {len(all_prices)}/{total} asset")
        else:
            with _DL_LOCK:
                _DL_STATE['status'] = 'error'
            print("❌ Download fallito: nessun dato disponibile")

    except Exception as e:
        print(f"❌ Download worker crash: {e}")
        with _DL_LOCK:
            _DL_STATE['status'] = 'error'


# ─────────────────────────────────────────────────────────────────────────────
# Carica solo nomi (avvio rapido)
# ─────────────────────────────────────────────────────────────────────────────
def load_ticker_names_only():
    try:
        df          = pd.read_excel(_XLSX)
        col_names   = df.columns.tolist()
        if len(col_names) < 3:
            return [], {}
        tickers     = list(df[col_names[0]])
        descrizione = list(df[col_names[1]])
        ticker_map  = {descrizione[i]: tickers[i] for i in range(len(tickers))}
        options     = [{'label': d, 'value': d} for d in descrizione]
        print(f"✓ Nomi caricati: {len(options)} asset")
        return options, ticker_map
    except Exception as e:
        print(f"Errore lettura nomi: {e}")
        return [], {}


# ─────────────────────────────────────────────────────────────────────────────
# Gestione sessioni
# ─────────────────────────────────────────────────────────────────────────────
SESSIONS_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions'))
SESSIONS_DIR.mkdir(exist_ok=True)
_INDEX_FILE = SESSIONS_DIR / 'index.json'

CLIENT_SESSION_STORES = [
    "weights-store-P1",
    "weights-store-P2",
    "weights-store-P3",
    "global-assets-selected",
    "stock-data",
    "original-prices-data",
    "asset-checklist",
    "ticker-map-store",
    "insufficient-data-store",
]


def _read_index():
    if not _INDEX_FILE.exists():
        return []
    try:
        return json.loads(_INDEX_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []

def _write_index(records):
    _INDEX_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

def list_sessions():
    records = _read_index()
    return sorted(records, key=lambda r: r.get('updated_at', ''), reverse=True)

def save_session(name, description, store_data):
    sid  = str(uuid.uuid4())[:8]
    now  = datetime.now().isoformat(timespec='seconds')
    data = json.dumps(store_data, ensure_ascii=False)
    size_kb = round(len(data.encode()) / 1024, 1)
    (SESSIONS_DIR / f'{sid}.json').write_text(data, encoding='utf-8')
    rec = {'id': sid, 'name': name or sid, 'description': description,
           'size_kb': size_kb, 'created_at': now, 'updated_at': now}
    records = _read_index()
    records.append(rec)
    _write_index(records)
    return rec

def load_session(session_id):
    f = SESSIONS_DIR / f'{session_id}.json'
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        return {}

def delete_session(session_id):
    f = SESSIONS_DIR / f'{session_id}.json'
    if f.exists():
        f.unlink()
    records = [r for r in _read_index() if r['id'] != session_id]
    _write_index(records)


def _format_ts(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime('%d/%m/%Y %H:%M')
    except Exception:
        return iso_str[:16]


def _build_session_row(record):
    sid  = record['id']
    name = record.get('name', sid[:8])
    desc = record.get('description', '')
    ts   = _format_ts(record.get('updated_at', record.get('created_at', '')))
    size = record.get('size_kb', '?')
    btn_s = {'border': 'none', 'border-radius': '3px', 'cursor': 'pointer',
             'font-size': '10px', 'padding': '3px 8px', 'font-weight': 'bold'}
    return html.Div([
        html.Div([
            html.Span(name, style={'font-weight': 'bold', 'font-size': '11px',
                                   'color': '#1a3a5c', 'margin-right': '8px'}),
            html.Span(f'({size} KB)', style={'font-size': '9px', 'color': '#999'}),
            html.Br(),
            html.Span(desc[:60] + ('…' if len(desc) > 60 else ''),
                      style={'font-size': '10px', 'color': '#555'}),
            html.Br(),
            html.Span(f'🕐 {ts}', style={'font-size': '9px', 'color': '#888'}),
        ], style={'flex': '1', 'min-width': '0'}),
        html.Div([
            html.Button('📂 Carica',
                        id={'type': 'session-load-btn',   'index': sid},
                        n_clicks=0,
                        style={**btn_s, 'background-color': '#1a3a5c',
                               'color': 'white', 'margin-right': '4px'}),
            html.Button('🗑 Elimina',
                        id={'type': 'session-delete-btn', 'index': sid},
                        n_clicks=0,
                        style={**btn_s, 'background-color': '#c0392b', 'color': 'white'}),
        ], style={'display': 'flex', 'align-items': 'center',
                  'flex-shrink': '0', 'margin-left': '10px'}),
    ], style={'display': 'flex', 'align-items': 'center', 'padding': '6px 8px',
              'border-bottom': '1px solid #eee', 'background': 'white',
              'border-radius': '3px', 'margin-bottom': '3px'})


def get_session_panel_layout():
    btn_base = {'border': 'none', 'border-radius': '4px', 'cursor': 'pointer',
                'font-size': '11px', 'padding': '4px 10px', 'font-weight': 'bold'}
    return html.Div([
        html.Button('💾 Sessioni', id='session-toggle-btn', n_clicks=0,
                    style={**btn_base, 'background-color': '#1a3a5c',
                           'color': 'white', 'margin-left': '12px',
                           'padding': '6px 14px', 'font-size': '12px'}),
        html.Div(id='session-panel', style={'display': 'none'}, children=[
            html.Div([
                # Colonna sinistra: salva
                html.Div([
                    html.B('💾 Salva sessione corrente',
                           style={'font-size': '12px', 'color': '#1a3a5c',
                                  'display': 'block', 'margin-bottom': '8px'}),
                    dcc.Input(id='session-name-input', type='text',
                              placeholder='Nome sessione…', debounce=False,
                              style={'width': '100%', 'margin-bottom': '6px',
                                     'padding': '5px 8px', 'border': '1px solid #aaa',
                                     'border-radius': '4px', 'font-size': '11px'}),
                    dcc.Textarea(id='session-desc-input',
                                 placeholder='Note / descrizione (opzionale)…', rows=2,
                                 style={'width': '100%', 'margin-bottom': '6px',
                                        'padding': '5px 8px', 'border': '1px solid #aaa',
                                        'border-radius': '4px', 'font-size': '11px',
                                        'resize': 'none'}),
                    html.Button('💾 Salva', id='session-save-btn', n_clicks=0,
                                style={**btn_base, 'background-color': '#1b7a34',
                                       'color': 'white', 'width': '100%'}),
                    html.Div(id='session-save-status',
                             style={'font-size': '10px', 'margin-top': '5px',
                                    'color': '#555', 'min-height': '16px'}),
                ], style={'width': '30%', 'padding-right': '20px',
                          'border-right': '1px solid #ddd'}),
                # Colonna destra: elenco
                html.Div([
                    html.Div([
                        html.B('📂 Sessioni salvate',
                               style={'font-size': '12px', 'color': '#1a3a5c'}),
                        html.Button('🔄', id='session-refresh-btn', n_clicks=0,
                                    title='Aggiorna elenco',
                                    style={**btn_base, 'background-color': '#e8e8e8',
                                           'color': '#333', 'margin-left': '8px',
                                           'padding': '3px 8px'}),
                    ], style={'display': 'flex', 'align-items': 'center',
                              'margin-bottom': '8px'}),
                    html.Div(id='session-list-container',
                             style={'max-height': '240px', 'overflow-y': 'auto'}),
                    dcc.Store(id='session-load-trigger',   data=None),
                    dcc.Store(id='session-delete-trigger', data=None),
                    dcc.Store(id='session-selected-id',    data=None),
                ], style={'flex': '1', 'padding-left': '20px'}),
            ], style={'display': 'flex'}),
        ]),
    ], style={'display': 'inline-block'})


# ─────────────────────────────────────────────────────────────────────────────
# Date range bar
# ─────────────────────────────────────────────────────────────────────────────
def get_date_range_bar(suffix):
    _today         = pd.Timestamp.today().normalize()
    _ten_years_ago = (_today - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    _today_str     = _today.strftime('%Y-%m-%d')
    return html.Div([
        html.Div([
            html.Div('📅 Range temporale:',
                     style={'font-size': '11px', 'font-weight': 'bold',
                            'color': '#1a3a5c', 'white-space': 'nowrap',
                            'margin-right': '10px', 'align-self': 'center'}),
            html.Div('da:', style={'font-size': '11px', 'color': '#555',
                                   'align-self': 'center', 'white-space': 'nowrap',
                                   'margin-right': '4px'}),
            dcc.DatePickerSingle(id=f'dr-start-{suffix}', display_format='DD/MM/YYYY',
                                 first_day_of_week=1, date=_ten_years_ago,
                                 clearable=False, style={'width': '135px'}),
            html.Div('a:', style={'font-size': '11px', 'color': '#555',
                                  'align-self': 'center', 'white-space': 'nowrap',
                                  'margin': '0 4px'}),
            dcc.DatePickerSingle(id=f'dr-end-{suffix}', display_format='DD/MM/YYYY',
                                 first_day_of_week=1, date=_today_str,
                                 clearable=False, style={'width': '135px'}),
            html.Div(id=f'dr-label-{suffix}',
                     style={'font-size': '10px', 'color': '#888', 'white-space': 'nowrap',
                            'align-self': 'center', 'margin-left': '10px',
                            'min-width': '200px'}),
        ], style={'display': 'flex', 'align-items': 'center', 'padding': '6px 12px',
                  'background': '#f0f4fa', 'border': '1px solid #d0d8e8',
                  'border-radius': '6px', 'margin-bottom': '8px', 'gap': '4px'}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Layout Tab 1
# ─────────────────────────────────────────────────────────────────────────────
def get_portfolio_analysis_tab(options_tickers):
    return html.Div([
        html.Div([
            # ── Riga controlli ────────────────────────────────────────────
            html.Div([
                html.Span('Rendimenti a confronto',
                          style={'margin-right': '16px', 'white-space': 'nowrap',
                                 'font-size': '13px', 'font-weight': '700',
                                 'color': '#1a3a5c'}),
                html.Div([
                    html.Div([
                        html.Label('Benchmark:',
                                   style={'margin-right': '6px', 'white-space': 'nowrap',
                                          'font-size': '11px'}),
                        dcc.Dropdown(
                            options=options_tickers or [],
                            value=None,
                            id='benchmark-selector',
                            placeholder='Seleziona…',
                            style={'width': '220px', 'font-size': '10px', 'min-width': '220px'}
                        ),
                    ], style={'margin-right': '16px', 'display': 'flex', 'align-items': 'center'}),
                    html.Div([
                        html.Label('AKRatio W:',
                                   style={'margin-right': '5px', 'white-space': 'nowrap',
                                          'font-size': '11px'}),
                        dcc.Input(id='ir-window-input', type='number', value=30, min=1,
                                  placeholder='30', style={'width': '55px', 'font-size': '11px'}),
                    ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '12px'}),
                    html.Div([
                        html.Label('Vol W:',
                                   style={'margin-right': '5px', 'white-space': 'nowrap',
                                          'font-size': '11px'}),
                        dcc.Input(id='vol-window-input', type='number', value=30, min=1,
                                  placeholder='30', style={'width': '55px', 'font-size': '11px'}),
                    ], style={'display': 'flex', 'align-items': 'center', 'margin-right': '12px'}),
                    html.Button('Update', id='update-portfolio-button', n_clicks=0,
                                style={'background-color': '#c0392b', 'color': 'white',
                                       'border': 'none', 'padding': '5px 14px',
                                       'border-radius': '4px', 'cursor': 'pointer',
                                       'font-weight': 'bold', 'font-size': '11px',
                                       'box-shadow': '0 2px 6px rgba(192,57,43,0.4)'}),
                    html.Div([
                        html.Label('Filtro AKRatio:',
                                   style={'font-size': '10px', 'font-weight': 'bold',
                                          'color': '#1a3a5c', 'margin-right': '5px',
                                          'white-space': 'nowrap'}),
                        dcc.RadioItems(
                            id='ir-filter-radio',
                            options=[
                                {'label': 'Tutti', 'value': 'all'},
                                {'label': '>−1',   'value': 'gt_minus1'},
                                {'label': '>0',    'value': 'gt_0'},
                            ],
                            value='all', inline=True,
                            inputStyle={'margin-right': '2px'},
                            labelStyle={'margin-right': '8px', 'font-size': '10px',
                                        'cursor': 'pointer'},
                        ),
                    ], style={'display': 'flex', 'align-items': 'center',
                              'margin-left': '12px', 'padding': '3px 8px',
                              'background': '#f0f4fa', 'border': '1px solid #d0d8e8',
                              'border-radius': '5px'}),
                ], style={'display': 'flex', 'align-items': 'center', 'flex-wrap': 'nowrap'}),
            ], style={'margin-bottom': '8px', 'display': 'flex', 'align-items': 'center',
                      'flex-wrap': 'nowrap', 'padding': '6px 8px',
                      'background': '#f8fafc', 'border': '1px solid #e2e8f0',
                      'border-radius': '6px'}),

            html.Hr(),

            # ── Intestazione full-width: etichette colonne (sx) + range date (dx) ──
            html.Div([
                # Sx 35%: etichette colonne — speculari alle colonne della griglia
                html.Div([
                    html.Div('Asset', **{'data-tooltip': 'Nome dell\'asset'}, style={
                        'width': '14%', 'font-weight': 'bold', 'font-size': '8px',
                        'padding-left': '5px', 'color': '#1a3a5c',
                        'display': 'flex', 'align-items': 'center',
                        'position': 'relative', 'cursor': 'default',
                    }),
                    *[html.Div(lbl, **{'data-tooltip': tip}, style={
                        'width': w, 'text-align': 'center', 'font-weight': 'bold',
                        'font-size': '8px', 'color': col,
                        'display': 'flex', 'align-items': 'center', 'justify-content': 'center',
                        'position': 'relative', 'cursor': 'default',
                    }) for w, lbl, col, tip in [
                        ('4%',  'CH',   '#1a3a5c', 'Grafici degli asset'),
                        ('8%',  'P1',   '#e6194b', 'Portafoglio 1'),
                        ('8%',  'P2',   '#3cb44b', 'Portafoglio 2'),
                        ('8%',  'P3',   '#4363d8', 'Portafoglio 3'),
                        ('5%',  'AKR',  '#1a3a5c', 'AKRatio — Extrarendimento sul benchmark per unità di rischio'),
                        ('8%',  'SH',   '#1a3a5c', 'Sharpe Ratio'),
                        ('8%',  'TV',   '#1a3a5c', 'Tracking Error Volatility'),
                        ('9%',  'DD',   '#1a3a5c', 'Draw Down'),
                        ('9%',  'VOL',  '#1a3a5c', 'Deviazione Standard'),
                        ('9%',  'VA90', '#1a3a5c', 'Value at Risk 90%'),
                        ('10%', 'VA95', '#1a3a5c', 'Value at Risk 95%'),
                    ]],
                ], style={
                    'width': '35%', 'display': 'flex', 'align-items': 'center',
                    'min-height': '32px', 'padding': '2px 0',
                }),
                # Dx 65%: range temporale — allineato a destra
                html.Div(
                    get_date_range_bar('tab1'),
                    style={'width': '65%', 'display': 'flex', 'align-items': 'center',
                           'justify-content': 'flex-end'},
                ),
            ], style={
                'display': 'flex', 'background': '#eaf4fb',
                'border-top': '2px solid #2e6da4', 'border-bottom': '1px solid #aed6f1',
            }),

            # ── Riga dati: griglia asset (sx 35%) + grafico (dx 65%) ─────
            html.Div([
                # Colonna sinistra: pulsanti Des + righe asset (generati dal callback)
                html.Div([
                    html.Div(id='asset-count-display',
                             style={'font-size': '10px', 'color': '#555',
                                    'padding': '3px 5px 5px 5px', 'margin-bottom': '2px'}),
                    html.Div(
                        '▶ Dati caricati — clicca UPDATE per aggiornare i grafici',
                        id='update-hint',
                        style={'display': 'none', 'font-size': '9px', 'color': '#c0392b',
                               'font-weight': '600', 'padding': '2px 5px 4px 5px',
                               'background': '#fdf2f0', 'border-left': '3px solid #c0392b',
                               'margin-bottom': '4px', 'border-radius': '0 4px 4px 0'}
                    ),
                    html.Div(id='weights-grid-container', style={'display': 'block'}),
                    html.Div([
                        html.Hr(style={'margin': '10px 0'}),
                        html.Div([
                            html.Div('Totale Pesi:',
                                     style={'width': '18%', 'font-weight': 'bold',
                                            'padding-left': '5px', 'font-size': '10px'}),
                            html.Div(id='sum-p1-display', children='0%',
                                     style={'width': '8%', 'text-align': 'center',
                                            'color': '#d62728', 'font-size': '10px'}),
                            html.Div(id='sum-p2-display', children='0%',
                                     style={'width': '8%', 'text-align': 'center',
                                            'color': '#d62728', 'font-size': '10px'}),
                            html.Div(id='sum-p3-display', children='0%',
                                     style={'width': '8%', 'text-align': 'center',
                                            'color': '#d62728', 'font-size': '10px'}),
                            html.Div('', style={'width': '58%'}),
                        ], style={'display': 'flex'}),
                    ], style={'margin-top': '10px'}),
                ], style={'width': '35%', 'vertical-align': 'top'}),

                # Colonna destra: grafico
                html.Div([
                    html.Div(
                        dcc.Graph(id='controls-and-graph',
                                  style={'width': '100%', 'height': '1900px',
                                         'margin': '0', 'padding': '0'},
                                  config={'responsive': True}),
                        style={'overflow-y': 'auto', 'max-height': '82vh',
                               'width': '100%', 'margin-bottom': '-10px'},
                    ),
                    html.Div(id='output', style={'display': 'none'}),
                    html.Div(id='date-values', style={'display': 'none'}),
                    dcc.Textarea(id='selected-column',           value='', style={'display': 'none'}),
                    dcc.Textarea(id='insufficient-data-tickers', value='', style={'display': 'none'}),
                ], style={'width': '65%', 'vertical-align': 'top'}),
            ], style={'display': 'flex', 'border-bottom': '2px solid #2e6da4'}),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Stili modale progresso
# ─────────────────────────────────────────────────────────────────────────────
_MODAL_HIDDEN = {
    'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0',
    'width': '100%', 'height': '100%', 'background': 'rgba(26,58,92,0.45)',
    'zIndex': '2000', 'justifyContent': 'center', 'alignItems': 'center',
}
_MODAL_SHOWN = {**_MODAL_HIDDEN, 'display': 'flex'}

_FILL_LOADING = {
    'height': '100%', 'width': '0%',
    'background': 'linear-gradient(90deg,#007755,#00aa77)',
    'borderRadius': '8px', 'transition': 'width 0.5s ease',
}
_STATUS_GREY  = {'fontSize': '0.78rem', 'color': '#6b7a99',
                 'fontFamily': 'Inter, sans-serif', 'textAlign': 'center',
                 'minHeight': '20px', 'marginTop': '6px'}
_STATUS_GREEN = {**_STATUS_GREY, 'color': '#007755', 'fontWeight': '600'}
_STATUS_RED   = {**_STATUS_GREY, 'color': '#c0392b', 'fontWeight': '600'}


# ─────────────────────────────────────────────────────────────────────────────
# Layout principale
# ─────────────────────────────────────────────────────────────────────────────
def _navbar():
    """Navbar identica al sito profilo, sticky in cima alla dashboard."""
    link_style = {
        'fontSize': '0.82rem', 'fontWeight': '600',
        'color': '#6b7a99', 'letterSpacing': '0.04em',
        'textTransform': 'uppercase', 'textDecoration': 'none',
        'transition': 'color 0.2s', 'fontFamily': 'Inter, sans-serif',
    }
    return html.Nav([
        # Brand
        html.A([
            html.Span('A·C', style={
                'fontFamily': "'Playfair Display', serif",
                'fontSize': '1.1rem', 'color': '#1a3a6b',
                'fontWeight': '700', 'marginRight': '10px',
            }),
            html.Span('FinecoBank', style={
                'fontFamily': 'Inter, sans-serif',
                'fontSize': '0.62rem', 'fontWeight': '700',
                'letterSpacing': '0.1em', 'textTransform': 'uppercase',
                'color': '#f37021',
                'background': 'rgba(243,112,33,0.1)',
                'border': '1px solid rgba(243,112,33,0.3)',
                'padding': '3px 8px', 'borderRadius': '4px',
            }),
        ], href='https://andcappe.github.io', target='_blank',
           style={'textDecoration': 'none', 'display': 'flex', 'alignItems': 'center'}),

        # Link di navigazione
        html.Ul([
            html.Li(html.A('Chi Sono',     href='https://andcappe.github.io#chi-sono',   target='_blank', style=link_style)),
            html.Li(html.A('Esperienza',   href='https://andcappe.github.io#esperienza', target='_blank', style=link_style)),
            html.Li(html.A('Strumenti',    href='https://andcappe.github.io#dashboard',  target='_blank', style=link_style)),
            html.Li(html.A('Prenota Call', href='https://andcappe.github.io#prenota',    target='_blank', style=link_style)),
            html.Li(html.A('Contatti',     href='https://andcappe.github.io#contatti',   target='_blank', style=link_style)),
        ], style={'display': 'flex', 'gap': '2rem', 'listStyle': 'none',
                  'margin': '0', 'padding': '0', 'alignItems': 'center'}),

        # CTA
        html.A([
            html.I(className='fa-regular fa-calendar', style={'marginRight': '7px'}),
            'Prenota call',
        ], href='https://andcappe.github.io#prenota', target='_blank', style={
            'padding': '9px 20px',
            'background': '#1a3a6b', 'color': 'white',
            'borderRadius': '7px', 'fontSize': '0.8rem', 'fontWeight': '700',
            'letterSpacing': '0.04em', 'textTransform': 'uppercase',
            'textDecoration': 'none', 'display': 'inline-flex',
            'alignItems': 'center', 'fontFamily': 'Inter, sans-serif',
        }),
    ], style={
        'position': 'fixed', 'top': '0', 'left': '0', 'right': '0',
        'zIndex': '1000',
        'display': 'flex', 'alignItems': 'center',
        'justifyContent': 'space-between',
        'padding': '0 5%', 'height': '64px',
        'background': 'rgba(255,255,255,0.96)',
        'backdropFilter': 'blur(14px)',
        'borderBottom': '1px solid #e2e8f0',
        'boxShadow': '0 1px 8px rgba(26,58,107,0.07)',
        'fontFamily': 'Inter, sans-serif',
    })


app.layout = html.Div([
    # ── Navbar ───────────────────────────────────────────────────────────────
    _navbar(),

    # ── Contenuto (margine top per navbar fissa 64px) ────────────────────────
    html.Div([

    # ── Intestazione pagina ───────────────────────────────────────────────────
    html.Div([
        html.H1('Analisi di Portafoglio', style={
            'margin': '0',
            'font-size': '1.6rem',
            'font-weight': '700',
            'color': '#1a3a6b',
            'font-family': "'Playfair Display', serif",
            'letter-spacing': '0.02em',
        }),
        html.P('Analisi quantitativa del portafoglio', style={
            'margin': '2px 0 0 0',
            'font-size': '0.78rem',
            'color': '#6b7a99',
            'font-family': 'Inter, sans-serif',
        }),
    ], style={
        'padding': '14px 20px 12px',
        'border-bottom': '2px solid #e2e8f0',
        'background': 'linear-gradient(90deg, #f0f4fb 0%, #ffffff 100%)',
        'margin-bottom': '10px',
    }),

    # ── Barra comandi ─────────────────────────────────────────────────────────
    html.Div([
        # 1. Carica Dati
        dcc.Loading(type='circle', color='#007755', children=[
            html.Button('▶ Carica Dati', id='load-data-button', n_clicks=0,
                        style={'background-color': '#007755', 'color': 'white',
                               'border': 'none', 'padding': '7px 16px',
                               'border-radius': '4px', 'cursor': 'pointer',
                               'font-weight': 'bold', 'font-size': '12px',
                               'margin-right': '8px'}),
        ]),
        # 2. Sessioni
        get_session_panel_layout(),
        # 3. Upload
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Trascina il tuo file']),
            style={'width': '150px', 'height': '32px', 'lineHeight': '32px',
                   'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                   'textAlign': 'center', 'margin': '0 8px', 'font-size': '11px',
                   'color': '#555', 'cursor': 'pointer'},
            multiple=False,
        ),
        # 4. Salva
        html.Button('💾 Salva', id='save-data-button', n_clicks=0,
                    style={'font-size': '11px', 'padding': '5px 12px',
                           'border-radius': '4px', 'cursor': 'pointer',
                           'background': '#f0f4fb', 'border': '1px solid #c0d0e8',
                           'color': '#1a3a5c', 'margin-right': '8px'}),
        # delete-column-button: nascosto ma necessario per il callback
        html.Button(id='delete-column-button', n_clicks=0,
                    style={'display': 'none'}),
        # Indicatori progresso (compatti)
        html.Div([
            html.Div(id='dl-progress-text',
                     style={'font-size': '10px', 'color': '#007755'}),
            html.Div(html.Div(id='dl-progress-fill',
                              style={'height': '4px', 'background': '#007755',
                                     'border-radius': '2px', 'width': '0%',
                                     'transition': 'width 0.4s ease'}),
                     id='dl-progress-bar-container',
                     style={'width': '120px', 'background': '#ddd',
                            'border-radius': '2px', 'display': 'none',
                            'margin-top': '2px'}),
        ]),
        # Loading invisibili (servono ai callback)
        html.Div(id='download-status',  style={'display': 'none'}),
        html.Div(id='upload-status',    style={'display': 'none'}),
    ], style={'display': 'flex', 'align-items': 'center',
              'font-size': '10px', 'position': 'relative',
              'padding': '6px 0', 'flex-wrap': 'wrap', 'gap': '2px'}),

    # ── Stores ───────────────────────────────────────────────────────────────
    dcc.Interval(id='dl-poll-interval', interval=800, n_intervals=0, disabled=True),
    dcc.Store(id='asset-checklist',         data=[]),
    dcc.Store(id='data-loaded-flag',        data=False),
    dcc.Store(id='stock-data',              data=None),
    dcc.Store(id='original-prices-data',    data=None),
    dcc.Store(id='ticker-map-store',        data={}),
    dcc.Store(id='insufficient-data-store', data=[]),
    dcc.Store(id='weights-store-P1',        data={}),
    dcc.Store(id='weights-store-P2',        data={}),
    dcc.Store(id='weights-store-P3',        data={}),
    dcc.Store(id='global-assets-selected',  data=[]),
    dcc.Store(id='tab1-slider-store',       data=None),
    dcc.Download(id='download-data'),

    # ── Modale progresso download ─────────────────────────────────────────────
    html.Div([
        html.Div([
            # Intestazione
            html.Div([
                html.Div([
                    html.I(className='fas fa-chart-line',
                           style={'marginRight': '8px', 'color': '#007755'}),
                    html.Span('Caricamento Dati di Mercato', style={
                        'fontFamily': "'Playfair Display', serif",
                        'fontSize': '1.05rem', 'fontWeight': '700',
                        'color': '#1a3a5c',
                    }),
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Button('✕', id='progress-modal-close', n_clicks=0, style={
                    'background': 'none', 'border': 'none', 'cursor': 'pointer',
                    'fontSize': '18px', 'color': '#6b7a99', 'padding': '0 4px',
                    'lineHeight': '1', 'marginLeft': '16px',
                }),
            ], style={
                'display': 'flex', 'justifyContent': 'space-between',
                'alignItems': 'center', 'marginBottom': '22px',
                'paddingBottom': '14px', 'borderBottom': '1px solid #e2e8f0',
            }),
            # Barra progresso
            html.Div(
                html.Div(id='modal-progress-fill', style=_FILL_LOADING),
                style={
                    'width': '100%', 'height': '12px', 'background': '#e2e8f0',
                    'borderRadius': '8px', 'overflow': 'hidden', 'marginBottom': '12px',
                },
            ),
            # Testo percentuale
            html.Div(id='modal-pct-text', style={
                'fontSize': '0.9rem', 'color': '#1a3a5c', 'fontWeight': '700',
                'fontFamily': 'Inter, sans-serif', 'textAlign': 'center',
                'marginBottom': '6px',
            }),
            # Messaggio stato
            html.Div(id='modal-status-text', style=_STATUS_GREY),
        ], style={
            'background': '#ffffff', 'borderRadius': '14px',
            'padding': '30px 36px', 'width': '440px', 'maxWidth': '90vw',
            'boxShadow': '0 24px 64px rgba(26,58,92,0.22)',
        }),
    ], id='progress-modal-overlay', style=_MODAL_HIDDEN),

    # ── Contenuto Tab 1 ───────────────────────────────────────────────────────
    html.Div(id='tab1-content'),
    ], style={'marginTop': '64px', 'padding': '0 1%'}),  # offset navbar fissa
])


# ─────────────────────────────────────────────────────────────────────────────
# Callback: inizializzazione + upload
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('upload-status',          'children'),
    Output('asset-checklist',        'data'),
    Output('stock-data',             'data'),
    Output('original-prices-data',   'data'),
    Output('insufficient-data-store', 'data'),
    Output('ticker-map-store',       'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('stock-data',  'data'),
)
def update_output(contents, filename, existing_stock_data):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'

    if triggered_id == 'initial_load':
        options, ticker_map = load_ticker_names_only()
        if options:
            return (
                html.Div('✓ Nomi caricati — clicca ▶ Carica Dati per scaricare i prezzi',
                         style={'color': '#007755', 'font-size': '11px'}),
                options, None, None, [], ticker_map
            )
        return html.Div('Nessun file trovato'), [], None, None, [], {}

    elif triggered_id == 'upload-data' and contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_excel(io.BytesIO(decoded))
            col_names   = df.columns.tolist()
            tickers     = list(df[col_names[0]])
            descrizione = list(df[col_names[1]])
            valuta      = list(df[col_names[2]])
            ticker_map  = {descrizione[i]: tickers[i] for i in range(len(tickers))}
            options     = [{'label': d, 'value': d} for d in descrizione]
            return (
                html.Div(f'✓ File caricato: {len(options)} asset — clicca ▶ Carica Dati',
                         style={'color': '#007755', 'font-size': '11px'}),
                options, None, None, [], ticker_map
            )
        except Exception as e:
            return html.Div(f'Errore: {e}'), [], None, None, [], {}

    raise PreventUpdate


# ─────────────────────────────────────────────────────────────────────────────
# Callback: renderizza contenuto Tab1
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('tab1-content', 'children'),
    Input('asset-checklist', 'data'),
)
def render_tab1(options_tickers):
    return get_portfolio_analysis_tab(options_tickers)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: avvia download
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('dl-poll-interval',        'disabled',    allow_duplicate=True),
    Output('dl-poll-interval',        'n_intervals', allow_duplicate=True),
    Output('load-data-button',        'disabled',    allow_duplicate=True),
    Output('data-loaded-flag',        'data',        allow_duplicate=True),
    Output('progress-modal-overlay',  'style',       allow_duplicate=True),
    Output('modal-progress-fill',     'style',       allow_duplicate=True),
    Output('modal-pct-text',          'children',    allow_duplicate=True),
    Output('modal-status-text',       'children',    allow_duplicate=True),
    Output('modal-status-text',       'style',       allow_duplicate=True),
    Input('load-data-button', 'n_clicks'),
    State('dr-start-tab1',    'date'),
    prevent_initial_call=True,
)
def start_download(n_clicks, start_date_picker):
    if not n_clicks:
        raise PreventUpdate
    try:
        df       = pd.read_excel(_XLSX)
        cols     = df.columns.tolist()
        tickers  = list(df[cols[0]])
        descr    = list(df[cols[1]])
        valuta   = list(df[cols[2]])
    except Exception as e:
        print(f"❌ Errore lettura Excel: {e}")
        err_fill = {**_FILL_LOADING, 'width': '100%', 'background': '#c0392b'}
        return (no_update, no_update, False, False,
                _MODAL_SHOWN, err_fill,
                'Errore', f'❌ Impossibile leggere il file Excel: {e}', _STATUS_RED)

    # Usa la data di inizio dal picker (default 10 anni fa)
    start_date = start_date_picker or (
        pd.Timestamp.today() - pd.DateOffset(years=10)
    ).strftime('%Y-%m-%d')

    t = threading.Thread(target=_download_worker,
                         args=(tickers, descr, valuta, start_date), daemon=True)
    t.start()
    print(f"▶ Thread download avviato: {len(tickers)} ticker da {start_date}")
    return (False, 0, True, False,
            _MODAL_SHOWN, _FILL_LOADING,
            f'Avvio download — {len(tickers)} asset da {start_date}…', '', _STATUS_GREY)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: polling progress
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('dl-progress-text',           'children'),
    Output('dl-progress-fill',           'style'),
    Output('dl-progress-bar-container',  'style'),
    Output('stock-data',                 'data',     allow_duplicate=True),
    Output('original-prices-data',       'data',     allow_duplicate=True),
    Output('asset-checklist',            'data',     allow_duplicate=True),
    Output('ticker-map-store',           'data',     allow_duplicate=True),
    Output('data-loaded-flag',           'data',     allow_duplicate=True),
    Output('dl-poll-interval',           'disabled', allow_duplicate=True),
    Output('load-data-button',           'disabled', allow_duplicate=True),
    Output('modal-progress-fill',        'style',    allow_duplicate=True),
    Output('modal-pct-text',             'children', allow_duplicate=True),
    Output('modal-status-text',          'children', allow_duplicate=True),
    Output('modal-status-text',          'style',    allow_duplicate=True),
    Input('dl-poll-interval', 'n_intervals'),
    prevent_initial_call=True,
)
def poll_download_progress(n):
    with _DL_LOCK:
        state  = dict(_DL_STATE)
        buffer = dict(_DL_BUFFER)

    status  = state.get('status', 'idle')
    current = state.get('current', 0)
    total   = state.get('total', 1) or 1
    pct     = int(current / total * 100)

    bar_style = {'height': '6px', 'background': '#007755', 'border-radius': '3px',
                 'width': f'{pct}%', 'transition': 'width 0.4s ease'}
    container_show = {'width': '180px', 'background': '#ddd',
                      'border-radius': '3px', 'display': 'block'}
    modal_fill = {**_FILL_LOADING, 'width': f'{pct}%'}

    if status == 'idle':
        raise PreventUpdate

    if status == 'running':
        return (f'Caricamento {current}/{total} ({pct}%)', bar_style, container_show,
                no_update, no_update, no_update, no_update, no_update,
                False, True,
                modal_fill, f'{current} / {total}  ({pct}%)', 'Download in corso…', _STATUS_GREY)

    if status == 'error':
        err_fill = {**_FILL_LOADING, 'width': '100%', 'background': '#c0392b'}
        return ('❌ Errore download', bar_style, container_show,
                no_update, no_update, no_update, no_update, False, True, False,
                err_fill, '❌ Download fallito',
                'Si è verificato un errore durante il download dei dati.', _STATUS_RED)

    close_returns   = buffer.get('close_returns')
    original_prices = buffer.get('original_prices')
    if close_returns is None or close_returns.empty:
        err_fill = {**_FILL_LOADING, 'width': '100%', 'background': '#c0392b'}
        return ('❌ Nessun dato', bar_style, container_show,
                no_update, no_update, no_update, no_update, False, True, False,
                err_fill, '❌ Nessun dato ricevuto',
                'Il download è terminato ma non sono stati restituiti dati.', _STATUS_RED)

    options = [{'label': col, 'value': col} for col in close_returns.columns]
    try:
        df        = pd.read_excel(_XLSX)
        col_names = df.columns.tolist()
        ticker_map = {list(df[col_names[1]])[i]: list(df[col_names[0]])[i]
                      for i in range(len(df))}
    except Exception:
        ticker_map = {}

    returns_json = close_returns.to_json(date_format='iso', orient='split')
    prices_json  = original_prices.to_json(date_format='iso', orient='split')
    ok_fill      = {**_FILL_LOADING, 'width': '100%'}

    return (
        f'✓ {len(options)} asset caricati',
        {**bar_style, 'width': '100%'},
        container_show,
        returns_json, prices_json, options, ticker_map, True, True, False,
        ok_fill, f'✓ {len(options)} asset caricati',
        'Dati pronti — puoi chiudere questa finestra e selezionare gli asset.', _STATUS_GREEN,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Callback: chiudi modale progresso
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('progress-modal-overlay', 'style', allow_duplicate=True),
    Input('progress-modal-close', 'n_clicks'),
    prevent_initial_call=True,
)
def close_progress_modal(n):
    if not n:
        raise PreventUpdate
    return _MODAL_HIDDEN


# ─────────────────────────────────────────────────────────────────────────────
# Callback: salva dati (download Excel)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('download-data',   'data'),
    Output('download-status', 'children'),
    Input('save-data-button', 'n_clicks'),
    State('stock-data',       'data'),
)
def salva_dati(n_clicks, stock_data):
    if n_clicks and n_clicks > 0 and stock_data:
        close_returns = _get_df(stock_data)
        try:
            buf = io.BytesIO()
            close_returns.to_excel(buf)
            buf.seek(0)
            return (
                dcc.send_bytes(buf.read(), 'rendimenti.xlsx'),
                html.Div('✓ File scaricato', style={'color': 'green', 'font-size': '11px'}),
            )
        except Exception as e:
            return no_update, html.Div(f'Errore: {e}', style={'color': 'red'})
    raise PreventUpdate


# ─────────────────────────────────────────────────────────────────────────────
# Callback: benchmark dropdown popolato dopo caricamento dati
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('benchmark-selector', 'options'),
    Output('benchmark-selector', 'value'),
    Input('asset-checklist', 'data'),
    State('benchmark-selector', 'value'),
)
def update_benchmark_options(options_tickers, current_value):
    if not options_tickers:
        return [], None
    if current_value and any(opt['value'] == current_value for opt in options_tickers):
        return options_tickers, current_value
    return options_tickers, options_tickers[0]['value'] if options_tickers else None


# ─────────────────────────────────────────────────────────────────────────────
# Callback: mostra/nascondi hint Update
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('update-hint', 'style'),
    Input('data-loaded-flag',        'data'),
    Input('update-portfolio-button', 'n_clicks'),
)
def toggle_update_hint(data_loaded, update_clicks):
    _shown  = {'display': 'block', 'font-size': '9px', 'color': '#c0392b',
                'font-weight': '600', 'padding': '2px 5px 4px 5px',
                'background': '#fdf2f0', 'border-left': '3px solid #c0392b',
                'margin-bottom': '4px', 'border-radius': '0 4px 4px 0'}
    _hidden = {**_shown, 'display': 'none'}
    ctx = callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
    if triggered == 'update-portfolio-button' and update_clicks:
        return _hidden
    if triggered == 'data-loaded-flag' and data_loaded:
        return _shown
    return _hidden


# ─────────────────────────────────────────────────────────────────────────────
# Callback: griglia pesi e asset
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('weights-grid-container', 'children'),
    Output('asset-count-display',    'children'),
    Input('update-portfolio-button', 'n_clicks'),
    State('data-loaded-flag',        'data'),
    State('stock-data',    'data'),
    State('asset-checklist', 'data'),
    State({'type': 'graph-select-checkbox',  'index': ALL}, 'value'),
    State({'type': 'ir-select-checkbox',     'index': ALL}, 'value'),
    State({'type': 'sharpe-select-checkbox', 'index': ALL}, 'value'),
    State({'type': 'tev-select-checkbox',    'index': ALL}, 'value'),
    State({'type': 'dd-select-checkbox',     'index': ALL}, 'value'),
    State({'type': 'vol-select-checkbox',    'index': ALL}, 'value'),
    State({'type': 'var90-select-checkbox',  'index': ALL}, 'value'),
    State({'type': 'var95-select-checkbox',  'index': ALL}, 'value'),
    State('weights-store-P1', 'data'),
    State('weights-store-P2', 'data'),
    State('weights-store-P3', 'data'),
    State('benchmark-selector', 'value'),
    State('ir-window-input',    'value'),
    State('ir-filter-radio',    'value'),
    prevent_initial_call=True,
)
def generate_asset_and_weight_inputs(update_clicks, data_loaded_flag, stock_data_json, options_tickers,
                                      graph_vals, ir_vals, sharpe_vals, tev_vals,
                                      dd_vals, vol_vals, var90_vals, var95_vals,
                                      saved_p1, saved_p2, saved_p3,
                                      benchmark_value, ir_window, ir_filter):
    _placeholder = html.Div(
        'Carica i dati per visualizzare gli asset',
        style={'color': '#888', 'font-style': 'italic', 'font-size': '11px', 'padding': '12px 8px'}
    )
    if not update_clicks:
        return [_placeholder], ''
    if not stock_data_json or not options_tickers:
        return [_placeholder], ''

    # Ripristina selezioni precedenti
    saved_selected = []
    for grp in (graph_vals or []) + (ir_vals or []) + (sharpe_vals or []) + \
               (tev_vals or []) + (dd_vals or []) + (vol_vals or []) + \
               (var90_vals or []) + (var95_vals or []):
        saved_selected.extend(grp)

    asset_names = [option['value'] for option in options_tickers]
    saved_p1 = saved_p1 or {}
    saved_p2 = saved_p2 or {}
    saved_p3 = saved_p3 or {}

    # Calcola AKRatio per colorazione etichette
    assets_above_threshold = set()
    if ir_filter and ir_filter != 'all' and benchmark_value:
        df_all = _get_df(stock_data_json)
        window = ir_window if (ir_window and ir_window > 0) else 30
        if df_all is not None and benchmark_value in df_all.columns:
            threshold = 0.0 if ir_filter == 'gt_0' else -1.0
            benchmark_returns = df_all[benchmark_value].dropna()
            for asset in asset_names:
                if asset == benchmark_value or asset not in df_all.columns:
                    continue
                combined = pd.concat([df_all[asset].dropna(), benchmark_returns], axis=1).dropna()
                if len(combined) >= window:
                    ir_series = calculate_rolling_information_ratio(
                        combined.iloc[:, 0], combined.iloc[:, 1], window=window)
                    last_ir = ir_series.dropna()
                    if not last_ir.empty and last_ir.iloc[-1] > threshold:
                        assets_above_threshold.add(asset)

    n_total = len(asset_names)
    n_above = len(assets_above_threshold)
    if ir_filter and ir_filter != 'all':
        label = '> 0' if ir_filter == 'gt_0' else '> −1'
        count_text = [
            html.Span(f'{n_above}', style={'font-weight': 'bold', 'color': '#c0392b', 'font-size': '12px'}),
            html.Span(f' / {n_total} asset', style={'color': '#555'}),
            html.Span(f'  in rosso (AKRatio {label})', style={'color': '#c0392b', 'font-style': 'italic'}),
        ]
    else:
        count_text = [
            html.Span(f'{n_total}', style={'font-weight': 'bold', 'color': '#1a3a5c', 'font-size': '12px'}),
            html.Span(f' / {n_total} asset', style={'color': '#555'}),
        ]

    def _lbl1(w, txt, color='#1a3a5c'):
        return html.Div(txt, style={'width': w, 'text-align': 'center', 'font-weight': 'bold',
                                    'font-size': '8px', 'color': color, 'display': 'flex',
                                    'align-items': 'center', 'justify-content': 'center'})

    def _btn1(btn_id, w):
        return html.Div(
            html.Button(
                html.I(className='fas fa-circle-xmark'),
                id=btn_id, n_clicks=0,
                style={
                    'width': '18px', 'height': '18px',
                    'padding': '0', 'cursor': 'pointer',
                    'background': '#f8d7da', 'border': '1px solid #e8a0a8',
                    'border-radius': '4px', 'color': '#a01830',
                    'display': 'flex', 'align-items': 'center',
                    'justify-content': 'center', 'font-size': '11px',
                    'line-height': '1',
                }),
            style={'width': w, 'display': 'flex',
                   'align-items': 'center', 'justify-content': 'center'})

    def _emp1(w):
        return html.Div('', style={'width': w})

    header = html.Div([
        html.Div('Asset', style={'width': '14%', 'font-weight': 'bold', 'font-size': '8px',
                                  'padding-left': '5px', 'color': '#1a3a5c',
                                  'display': 'flex', 'align-items': 'center'}),
        _lbl1('4%',  'CH'),
        _lbl1('8%',  'P1',  '#e6194b'),
        _lbl1('8%',  'P2',  '#3cb44b'),
        _lbl1('8%',  'P3',  '#4363d8'),
        _lbl1('5%',  'AKR'),
        _lbl1('8%',  'SH'),
        _lbl1('8%',  'TV'),
        _lbl1('9%',  'DD'),
        _lbl1('9%',  'VOL'),
        _lbl1('9%',  'VA90'),
        _lbl1('10%', 'VA95'),
    ], style={'display': 'flex', 'padding': '4px 0 2px', 'background': '#eaf4fb',
              'border-top': '2px solid #2e6da4', 'border-bottom': '1px solid #aed6f1'})

    labels_row = html.Div([
        _emp1('14%'),
        _btn1('deselect-all-tickers', '4%'),
        _btn1('reset-p1-tab1',        '8%'),
        _btn1('reset-p2-tab1',        '8%'),
        _btn1('reset-p3-tab1',        '8%'),
        _btn1('deselect-all-ir',      '5%'),
        _btn1('deselect-all-sharpe',  '8%'),
        _btn1('deselect-all-tev',     '8%'),
        _btn1('deselect-all-dd',      '9%'),
        _btn1('deselect-all-vol',     '9%'),
        _btn1('deselect-all-var90',   '9%'),
        _btn1('deselect-all-var95',   '10%'),
    ], style={'display': 'flex', 'padding': '3px 0 5px', 'background': '#f5faff',
              'border-bottom': '2px solid #2e6da4', 'margin-bottom': '4px'})

    rows = [labels_row]

    def _chk(chk_id, opt_val, sel_val, w):
        return html.Div(
            dcc.Checklist(id=chk_id,
                          options=[{'label': '', 'value': opt_val}],
                          value=sel_val,
                          style={'flex-direction': 'row', 'font-size': '10px',
                                 'width': '100%', 'justify-content': 'center'}),
            style={'width': w, 'height': '30px', 'display': 'flex',
                   'align-items': 'center', 'justify-content': 'center'})

    for asset in asset_names:
        def create_weight_input(portfolio_index, a=asset):
            port_key  = f'P{portfolio_index}'
            saved_val = {1: saved_p1, 2: saved_p2, 3: saved_p3}[portfolio_index].get(a, 0)
            return dcc.Input(
                id={'type': 'weight-input', 'index': f'{port_key}-{a}'},
                type='number', value=saved_val, min=0, max=100, step=0.1, placeholder='0',
                style={'width': '90%', 'text-align': 'right', 'margin-bottom': '5px'}
            )

        asset_val   = [asset]                       if asset in saved_selected else []
        ir_val      = [f'{asset}_InformationRatio'] if f'{asset}_InformationRatio' in saved_selected else []
        sharpe_val  = [f'{asset}_Sharpe']           if f'{asset}_Sharpe'           in saved_selected else []
        tev_val     = [f'{asset}_TEV']              if f'{asset}_TEV'              in saved_selected else []
        dd_val      = [f'{asset}_DD']               if f'{asset}_DD'               in saved_selected else []
        vol_val     = [f'{asset}_Vol']              if f'{asset}_Vol'              in saved_selected else []
        var90_val   = [f'{asset}_VaR90']            if f'{asset}_VaR90'            in saved_selected else []
        var95_val   = [f'{asset}_VaR95']            if f'{asset}_VaR95'            in saved_selected else []
        _label_color = '#c0392b' if asset in assets_above_threshold else '#1a3a5c'

        row_content = html.Div([
            html.Div(
                html.Div(
                    html.B(asset, style={'color': _label_color}),
                    style={'overflow': 'hidden', 'white-space': 'nowrap',
                           'text-overflow': 'ellipsis', 'width': '100%'},
                ),
                **{'data-tooltip': asset},
                style={'width': '14%', 'height': '30px', 'display': 'flex',
                       'align-items': 'center', 'padding-left': '5px',
                       'font-size': '8px', 'overflow': 'visible',
                       'position': 'relative', 'cursor': 'default'},
            ),
            _chk({'type': 'graph-select-checkbox',  'index': asset}, asset,                       asset_val,  '4%'),
            html.Div(create_weight_input(1), style={'width': '8%'}),
            html.Div(create_weight_input(2), style={'width': '8%'}),
            html.Div(create_weight_input(3), style={'width': '8%'}),
            _chk({'type': 'ir-select-checkbox',     'index': asset}, f'{asset}_InformationRatio', ir_val,     '5%'),
            _chk({'type': 'sharpe-select-checkbox', 'index': asset}, f'{asset}_Sharpe',           sharpe_val, '8%'),
            _chk({'type': 'tev-select-checkbox',    'index': asset}, f'{asset}_TEV',              tev_val,    '8%'),
            _chk({'type': 'dd-select-checkbox',     'index': asset}, f'{asset}_DD',               dd_val,     '9%'),
            _chk({'type': 'vol-select-checkbox',    'index': asset}, f'{asset}_Vol',              vol_val,    '9%'),
            _chk({'type': 'var90-select-checkbox',  'index': asset}, f'{asset}_VaR90',            var90_val,  '9%'),
            _chk({'type': 'var95-select-checkbox',  'index': asset}, f'{asset}_VaR95',            var95_val, '10%'),
        ], style={'display': 'flex', 'border-bottom': '1px dotted #eee'})
        rows.append(row_content)

    # Righe portafogli P1/P2/P3
    def _pchk(chk_id, opt_val, sel_val, w):
        return html.Div(
            dcc.Checklist(id=chk_id,
                          options=[{'label': '', 'value': opt_val}],
                          value=sel_val,
                          style={'flex-direction': 'row', 'font-size': '10px',
                                 'width': '100%', 'justify-content': 'center'}),
            style={'width': w, 'height': '30px', 'display': 'flex',
                   'align-items': 'center', 'justify-content': 'center'})

    for portfolio_num in [1, 2, 3]:
        portfolio_name  = f'Port{portfolio_num}'
        port_val        = [portfolio_name]                              if portfolio_name                              in saved_selected else []
        ir_port_val     = [f'{portfolio_name}_InformationRatio']       if f'{portfolio_name}_InformationRatio'       in saved_selected else []
        sharpe_port_val = [f'{portfolio_name}_Sharpe']                 if f'{portfolio_name}_Sharpe'                 in saved_selected else []
        tev_port_val    = [f'{portfolio_name}_TEV']                    if f'{portfolio_name}_TEV'                    in saved_selected else []
        dd_port_val     = [f'{portfolio_name}_DD']                     if f'{portfolio_name}_DD'                     in saved_selected else []
        vol_port_val    = [f'{portfolio_name}_Vol']                    if f'{portfolio_name}_Vol'                    in saved_selected else []
        var90_port_val  = [f'{portfolio_name}_VaR90']                  if f'{portfolio_name}_VaR90'                  in saved_selected else []
        var95_port_val  = [f'{portfolio_name}_VaR95']                  if f'{portfolio_name}_VaR95'                  in saved_selected else []

        portfolio_row = html.Div([
            html.Div(html.B(portfolio_name, style={'color': '#0066cc'}),
                     **{'data-tooltip': portfolio_name},
                     style={'width': '14%', 'height': '30px', 'display': 'flex',
                            'align-items': 'center', 'padding-left': '5px', 'font-size': '10px',
                            'cursor': 'default'}),
            _pchk({'type': 'graph-select-checkbox',  'index': portfolio_name}, portfolio_name,                       port_val,        '4%'),
            html.Div('', style={'width': '8%'}),
            html.Div('', style={'width': '8%'}),
            html.Div('', style={'width': '8%'}),
            _pchk({'type': 'ir-select-checkbox',     'index': portfolio_name}, f'{portfolio_name}_InformationRatio', ir_port_val,     '5%'),
            _pchk({'type': 'sharpe-select-checkbox', 'index': portfolio_name}, f'{portfolio_name}_Sharpe',           sharpe_port_val, '8%'),
            _pchk({'type': 'tev-select-checkbox',    'index': portfolio_name}, f'{portfolio_name}_TEV',              tev_port_val,    '8%'),
            _pchk({'type': 'dd-select-checkbox',     'index': portfolio_name}, f'{portfolio_name}_DD',               dd_port_val,     '9%'),
            _pchk({'type': 'vol-select-checkbox',    'index': portfolio_name}, f'{portfolio_name}_Vol',              vol_port_val,    '9%'),
            _pchk({'type': 'var90-select-checkbox',  'index': portfolio_name}, f'{portfolio_name}_VaR90',            var90_port_val,  '9%'),
            _pchk({'type': 'var95-select-checkbox',  'index': portfolio_name}, f'{portfolio_name}_VaR95',            var95_port_val, '10%'),
        ], style={'display': 'flex', 'border-bottom': '1px dotted #eee',
                  'background-color': '#f0f0f0'})
        rows.append(portfolio_row)

    return rows, count_text


# ─────────────────────────────────────────────────────────────────────────────
# Callback: raccoglie asset selezionati nello store globale
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('global-assets-selected', 'data'),
    Input({'type': 'graph-select-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def collect_selected_assets(all_values):
    return [v[0] for v in all_values if v]


# ─────────────────────────────────────────────────────────────────────────────
# Callback: somme pesi
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('sum-p1-display', 'children'),
    Output('sum-p2-display', 'children'),
    Output('sum-p3-display', 'children'),
    Output('sum-p1-display', 'style'),
    Output('sum-p2-display', 'style'),
    Output('sum-p3-display', 'style'),
    Output('weights-store-P1', 'data'),
    Output('weights-store-P2', 'data'),
    Output('weights-store-P3', 'data'),
    Input({'type': 'weight-input', 'index': ALL}, 'value'),
    State({'type': 'weight-input', 'index': ALL}, 'id'),
    State('weights-store-P1', 'data'),
    State('weights-store-P2', 'data'),
    State('weights-store-P3', 'data'),
    prevent_initial_call=True,
)
def update_portfolio_weights(all_input_values, all_input_ids, p1_data, p2_data, p3_data):
    p1 = dict(p1_data or {})
    p2 = dict(p2_data or {})
    p3 = dict(p3_data or {})

    for val, inp_id in zip(all_input_values, all_input_ids):
        idx = inp_id['index']
        if idx.startswith('P1-'):
            p1[idx[3:]] = val or 0
        elif idx.startswith('P2-'):
            p2[idx[3:]] = val or 0
        elif idx.startswith('P3-'):
            p3[idx[3:]] = val or 0

    sum1 = sum(v for v in p1.values() if v)
    sum2 = sum(v for v in p2.values() if v)
    sum3 = sum(v for v in p3.values() if v)

    def _style(s):
        base = {'width': '8%', 'text-align': 'center', 'font-size': '10px'}
        base['color'] = '#1b7a34' if abs(s - 100) < 0.01 else '#d62728'
        return base

    return (f'{sum1:.1f}%', f'{sum2:.1f}%', f'{sum3:.1f}%',
            _style(sum1), _style(sum2), _style(sum3),
            p1, p2, p3)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: seleziona colonna dal click sul grafico
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('selected-column', 'value'),
    Input('update-portfolio-button', 'n_clicks'),
    Input('delete-column-button',    'n_clicks'),
    Input('controls-and-graph',      'clickData'),
    State('controls-and-graph',      'figure'),
    prevent_initial_call=True,
)
def update_selected_column(update_clicks, delete_clicks, clickData, figure):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
    if triggered_id in ('update-portfolio-button', 'delete-column-button'):
        return ''
    if triggered_id == 'controls-and-graph' and clickData and clickData['points']:
        curve_number = clickData['points'][0]['curveNumber']
        if figure and 'data' in figure and curve_number < len(figure['data']):
            return figure['data'][curve_number].get('name', '')
    return ''


# ─────────────────────────────────────────────────────────────────────────────
# Callback: date picker → tab1-slider-store (grafico reagisce al cambio date)
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('dr-start-tab1',     'date'),
    Output('dr-end-tab1',       'date'),
    Output('dr-label-tab1',     'children'),
    Output('tab1-slider-store', 'data'),
    Input('stock-data',         'data'),
    Input('dr-start-tab1',      'date'),
    Input('dr-end-tab1',        'date'),
    prevent_initial_call=False,
)
def sync_date_range(stock_data, start_date, end_date):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

    if not stock_data:
        raise PreventUpdate

    df = _get_df(stock_data)
    if df is None or df.empty:
        raise PreventUpdate

    ts_min = int(df.index.min().timestamp())
    ts_max = int(df.index.max().timestamp())

    if triggered_id == 'stock-data':
        # Inizializzazione: 10 anni fa → oggi, clampati al range dei dati
        today         = pd.Timestamp.today().normalize()
        ten_years_ago = today - pd.DateOffset(years=10)
        s = max(ts_min, int(ten_years_ago.timestamp()))
        e = min(ts_max, int(today.timestamp()))
        d_start = pd.Timestamp(s, unit='s').strftime('%Y-%m-%d')
        d_end   = pd.Timestamp(e, unit='s').strftime('%Y-%m-%d')
        label   = (f"{pd.Timestamp(s, unit='s').strftime('%d/%m/%Y')} — "
                   f"{pd.Timestamp(e, unit='s').strftime('%d/%m/%Y')}")
        return d_start, d_end, label, [s, e]

    # L'utente ha cambiato una delle due date: non riscrivere i picker (evita loop)
    try:
        s = int(pd.Timestamp(start_date).timestamp()) if start_date else ts_min
        e = int(pd.Timestamp(end_date).timestamp())   if end_date   else ts_max
    except Exception:
        s, e = ts_min, ts_max

    s = max(ts_min, min(s, ts_max))
    e = max(ts_min, min(e, ts_max))

    label = (f"{pd.Timestamp(s, unit='s').strftime('%d/%m/%Y')} — "
             f"{pd.Timestamp(e, unit='s').strftime('%d/%m/%Y')}")

    return no_update, no_update, label, [s, e]


# ─────────────────────────────────────────────────────────────────────────────
# Callback: aggiorna grafico principale
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('controls-and-graph',        'figure'),
    Output('date-values',               'children'),
    Output('insufficient-data-tickers', 'value'),

    Input('update-portfolio-button', 'n_clicks'),
    Input('delete-column-button',    'n_clicks'),
    Input('controls-and-graph',      'clickData'),

    State('tab1-slider-store',       'data'),
    State('global-assets-selected',  'data'),
    State('benchmark-selector',      'value'),
    State('ir-window-input',         'value'),
    State('weights-store-P1',        'data'),
    State('weights-store-P2',        'data'),
    State('weights-store-P3',        'data'),
    State({'type': 'ir-select-checkbox',     'index': ALL}, 'value'),
    State({'type': 'sharpe-select-checkbox', 'index': ALL}, 'value'),
    State({'type': 'tev-select-checkbox',    'index': ALL}, 'value'),
    State({'type': 'dd-select-checkbox',     'index': ALL}, 'value'),
    State({'type': 'vol-select-checkbox',    'index': ALL}, 'value'),
    State({'type': 'var90-select-checkbox',  'index': ALL}, 'value'),
    State({'type': 'var95-select-checkbox',  'index': ALL}, 'value'),
    State('vol-window-input',        'value'),
    State('insufficient-data-store', 'data'),
    State('selected-column',         'value'),
    State('stock-data',              'data'),
    prevent_initial_call=True,
)
def update_graph(update_clicks, delete_clicks, clickData, date_range, selected_assets,
                 benchmark_value, ir_window,
                 weights_p1_data, weights_p2_data, weights_p3_data,
                 all_ir_checkbox_values, all_sharpe_checkbox_values, all_tev_checkbox_values,
                 all_dd_checkbox_values, all_vol_checkbox_values,
                 all_var90_checkbox_values, all_var95_checkbox_values, vol_window,
                 insufficient_data, selected_column, stock_data):

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not update_clicks and not delete_clicks and not clickData:
        raise PreventUpdate

    if not stock_data:
        return {}, 'Nessun dato caricato', ''

    def _collect(vals):
        result = []
        for v in (vals or []):
            if v:
                result.extend(v)
        return result

    selected_irs    = _collect(all_ir_checkbox_values)
    selected_sharpe = _collect(all_sharpe_checkbox_values)
    selected_tev    = _collect(all_tev_checkbox_values)
    selected_dd     = _collect(all_dd_checkbox_values)
    selected_vol    = _collect(all_vol_checkbox_values)
    selected_var90  = _collect(all_var90_checkbox_values)
    selected_var95  = _collect(all_var95_checkbox_values)

    vol_window = vol_window if (vol_window and vol_window > 0) else 30
    ir_window  = ir_window  if (ir_window  and ir_window  > 0) else 30

    close_returns = _get_df(stock_data)

    if date_range and len(date_range) == 2:
        start_date  = pd.to_datetime(date_range[0], unit='s')
        end_date    = pd.to_datetime(date_range[1], unit='s')
        filtered_df = close_returns.loc[start_date:end_date]
    else:
        filtered_df = close_returns
        start_date  = close_returns.index.min()
        end_date    = close_returns.index.max()

    asset_columns      = list(filtered_df.columns)
    df_with_portfolios = filtered_df.copy()

    weights_data_map = {
        'Port1': weights_p1_data,
        'Port2': weights_p2_data,
        'Port3': weights_p3_data,
    }

    def calc_portfolio(df, name, weights_dict, asset_cols):
        weights_dict = weights_dict or {}
        total_w = sum(weights_dict.values())
        normalized = {}
        if total_w > 0:
            for a, w in weights_dict.items():
                if a in asset_cols and w and w > 0:
                    normalized[a] = w / 100.0
        port = pd.Series(0.0, index=df.index, name=name)
        if normalized:
            for a, w in normalized.items():
                if a in df.columns:
                    port += df[a] * w
        return port

    for p_name, w_dict in weights_data_map.items():
        df_with_portfolios[p_name] = calc_portfolio(
            df_with_portfolios, p_name, w_dict, asset_columns)

    if benchmark_value is None or benchmark_value not in df_with_portfolios.columns:
        benchmark_col = df_with_portfolios.columns[0] if not df_with_portfolios.empty else None
    else:
        benchmark_col = benchmark_value

    if benchmark_col is None:
        return {}, 'Nessun dato caricato', ''

    benchmark_returns = df_with_portfolios[benchmark_col]
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1

    # Calcola IR/Sharpe/TEV solo per gli asset effettivamente selezionati
    _need_ir     = {s.replace('_InformationRatio', '') for s in selected_irs    if s}
    _need_sharpe = {si.replace('_Sharpe', '')          for si in selected_sharpe if si}
    _need_tev    = {ti.replace('_TEV', '')             for ti in selected_tev    if ti}

    _extra = {}
    for col in _need_ir:
        if col in df_with_portfolios.columns and col != benchmark_col:
            _extra[f'{col}_InformationRatio'] = calculate_rolling_information_ratio(
                df_with_portfolios[col], benchmark_returns, window=ir_window)
    for col in _need_sharpe:
        if col in df_with_portfolios.columns:
            _extra[f'{col}_SharpeRatio'] = calculate_rolling_sharpe_ratio(
                df_with_portfolios[col], window=ir_window)
    for col in _need_tev:
        if col in df_with_portfolios.columns and col != benchmark_col:
            _extra[f'{col}_TEV'] = calculate_tracking_error_volatility(
                df_with_portfolios[col], benchmark_returns, window=ir_window)

    if _extra:
        df_final = pd.concat([df_with_portfolios,
                               pd.DataFrame(_extra, index=df_with_portfolios.index)], axis=1)
    else:
        df_final = df_with_portfolios

    fig = make_subplots(
        rows=7, cols=1, shared_xaxes=False, vertical_spacing=0.03,
        row_heights=[0.26, 0.12, 0.12, 0.12, 0.12, 0.13, 0.13],
        specs=[[{"secondary_y": False}]] * 7,
        subplot_titles=('Cumulative Returns', 'AKRatio', 'Sharpe Ratio',
                        'TEV', 'DrawDown', 'Volatilità', 'VaR (90% / 95%)')
    )

    for row in range(1, 8):
        fig.add_trace(
            go.Scatter(x=[df_final.index[0], df_final.index[-1]], y=[0, 0],
                       mode='lines', line=dict(color='rgba(0,0,0,0)'),
                       showlegend=False, hoverinfo='skip'),
            row=row, col=1)

    selected_assets = selected_assets or []

    # Subplot 1: Cumulative Returns
    if benchmark_col in selected_assets:
        bench_name = f'{benchmark_col} Cum. Returns'
        is_sel = (selected_column == bench_name)
        fig.add_trace(go.Scatter(x=df_final.index, y=benchmark_cumulative_returns,
                                  name=bench_name,
                                  line=dict(color='red', width=8 if is_sel else 2),
                                  legend='legend'), row=1, col=1)
        color_index = 1
    else:
        color_index = 0

    for col in selected_assets:
        if col in df_with_portfolios.columns and col != benchmark_col:
            series = df_with_portfolios[col]
            if col.startswith('Port'):
                pnum    = int(col[-1])
                w_dict  = {1: weights_p1_data, 2: weights_p2_data, 3: weights_p3_data}.get(pnum, {}) or {}
                active  = [a for a, w in w_dict.items() if w and w > 0 and a in filtered_df.columns]
                if active:
                    first_valid = filtered_df[active].dropna(how='any').index.min()
                    if pd.notna(first_valid):
                        series = series.loc[first_valid:]
            cum_ret    = (1 + series).cumprod() - 1
            trace_name = f'{col} Cum. Returns'
            is_sel     = (selected_column == trace_name)
            if is_sel:
                line_dict = dict(color='red', width=8)
            else:
                tc = color_palette[color_index % len(color_palette)]
                line_dict = dict(color=tc)
                if col.startswith('Port'):
                    line_dict.update({'width': 4, 'dash': 'solid'})
            fig.add_trace(go.Scatter(x=series.index, y=cum_ret, name=trace_name,
                                      line=line_dict, legend='legend'), row=1, col=1)
            color_index += 1

    # Mappa colore per asset (evita scan O(n²) in ogni subplot successivo)
    _color_map = {}
    for _t in fig.data:
        _nm = _t.name or ''
        if ' Cum. Returns' in _nm:
            _asset = _nm.replace(' Cum. Returns', '')
            try:
                _c = _t.line.color
                if _c:
                    _color_map[_asset] = _c
            except AttributeError:
                pass

    # Subplot 2: AKRatio
    color_index = 0
    for col_ir in selected_irs:
        if col_ir in df_final.columns:
            orig = col_ir.replace('_InformationRatio', '')
            is_sel = (selected_column == col_ir)
            if is_sel:
                line_dict = dict(color='red', width=8)
            else:
                tc = _color_map.get(orig, color_palette[color_index % len(color_palette)])
                line_dict = dict(color=tc, width=4 if orig.startswith('Port') else 2.5)
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final[col_ir],
                                      name=col_ir.replace('_InformationRatio', '_AKRatio'),
                                      line=line_dict, legend='legend2'), row=2, col=1)
            color_index += 1
    if selected_irs:
        fig.add_trace(go.Scatter(x=df_final.index, y=[0]*len(df_final), name='Zero Line (IR)',
                                  line=dict(color='red', dash='solid', width=2),
                                  showlegend=True, legend='legend2'), row=2, col=1)

    # Subplot 3: Sharpe
    for si in selected_sharpe:
        if si:
            an = si.replace('_Sharpe', '')
            sc = f'{an}_SharpeRatio'
            if sc in df_final.columns:
                tn    = f'{an} Sharpe Ratio'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, dash='solid',
                                     width=4 if an.startswith('Port') else 2.5)
                fig.add_trace(go.Scatter(x=df_final.index, y=df_final[sc], name=tn,
                                          line=line_dict, legend='legend3'), row=3, col=1)

    # Subplot 4: TEV
    for ti in selected_tev:
        if ti:
            an = ti.replace('_TEV', '')
            tc_nm = f'{an}_TEV'
            if tc_nm in df_final.columns:
                tn    = f'{an} TEV'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, dash='dash',
                                     width=4 if an.startswith('Port') else 2.5)
                fig.add_trace(go.Scatter(x=df_final.index, y=df_final[tc_nm], name=tn,
                                          line=line_dict, legend='legend4'), row=4, col=1)

    # Subplot 5: DrawDown
    for di in selected_dd:
        if di:
            an = di.replace('_DD', '')
            if an in df_with_portfolios.columns:
                dds   = calculate_drawdown(df_with_portfolios[an])
                tn    = f'{an} DrawDown'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                    fillcolor = 'rgba(200,0,0,0.08)'
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, dash='dot',
                                     width=4 if an.startswith('Port') else 2.5)
                    fillcolor = (tc.replace('rgb', 'rgba').replace(')', ',0.10)')
                                 if tc and tc.startswith('rgb') else 'rgba(200,0,0,0.08)')
                fig.add_trace(go.Scatter(x=dds.index, y=dds, name=tn,
                                          line=line_dict, legend='legend5',
                                          fill='tozeroy', fillcolor=fillcolor), row=5, col=1)

    # Subplot 6: Volatilità
    for vi in selected_vol:
        if vi:
            an = vi.replace('_Vol', '')
            if an in df_with_portfolios.columns:
                vs    = _rolling_volatility(df_with_portfolios[an], vol_window)
                tn    = f'{an} Volatilità'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, width=4 if an.startswith('Port') else 2.5)
                fig.add_trace(go.Scatter(x=vs.index, y=vs, name=tn,
                                          line=line_dict, legend='legend6'), row=6, col=1)

    # Subplot 7: VaR
    for v90 in selected_var90:
        if v90:
            an = v90.replace('_VaR90', '')
            if an in df_with_portfolios.columns:
                vs    = calculate_rolling_cvar(df_with_portfolios[an], vol_window, 0.10)
                tn    = f'{an} VaR90'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, width=4 if an.startswith('Port') else 2.5,
                                     dash='solid')
                fig.add_trace(go.Scatter(x=vs.index, y=vs, name=tn,
                                          line=line_dict, legend='legend7'), row=7, col=1)

    for v95 in selected_var95:
        if v95:
            an = v95.replace('_VaR95', '')
            if an in df_with_portfolios.columns:
                vs    = calculate_rolling_cvar(df_with_portfolios[an], vol_window, 0.05)
                tn    = f'{an} VaR95'
                is_sel = (selected_column == tn)
                if is_sel:
                    line_dict = dict(color='red', width=8)
                else:
                    tc = _color_map.get(an, color_palette[color_index % len(color_palette)])
                    line_dict = dict(color=tc, width=4 if an.startswith('Port') else 2.5,
                                     dash='dot')
                fig.add_trace(go.Scatter(x=vs.index, y=vs, name=tn,
                                          line=line_dict, legend='legend7'), row=7, col=1)

    for row in range(1, 8):
        fig.update_xaxes(title_text='', row=row, col=1)
    fig.update_yaxes(title_text='Cumulative Returns', row=1, col=1)
    fig.update_yaxes(title_text='AKRatio',            row=2, col=1)
    fig.update_yaxes(title_text='Sharpe Ratio',       row=3, col=1)
    fig.update_yaxes(title_text='TEV',                row=4, col=1)
    fig.update_yaxes(title_text='DrawDown',           row=5, col=1)
    fig.update_yaxes(title_text='Volatilità (ann.)',  row=6, col=1)
    fig.update_yaxes(title_text='VaR',                row=7, col=1)

    fig.update_layout(
        height=1900, showlegend=True,
        legend=dict(title=dict(text='<b>Asset</b>', font=dict(size=11)),
                    orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.01,
                    bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend2=dict(title=dict(text='<b>AKRatio</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.72, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend3=dict(title=dict(text='<b>Sharpe Ratio</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.58, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend4=dict(title=dict(text='<b>TEV</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.44, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend5=dict(title=dict(text='<b>DrawDown</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.30, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend6=dict(title=dict(text='<b>Volatilità</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.16, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        legend7=dict(title=dict(text='<b>VaR — (solido 90%, punteg. 95%)</b>', font=dict(size=11)),
                     orientation='v', yanchor='top', y=0.02, xanchor='left', x=1.01,
                     bgcolor='rgba(255,255,255,0.85)', bordercolor='#aed6f1', borderwidth=1),
        margin=dict(b=20, t=60, r=220),
        autosize=True,
        xaxis=dict(range=[start_date, end_date]),
        xaxis2=dict(range=[start_date, end_date]),
        xaxis3=dict(range=[start_date, end_date]),
        xaxis4=dict(range=[start_date, end_date]),
        xaxis5=dict(range=[start_date, end_date]),
        xaxis6=dict(range=[start_date, end_date]),
        xaxis7=dict(range=[start_date, end_date]),
    )

    date_values = (f"Intervallo di date: {start_date.strftime('%d-%m-%Y')} — "
                   f"{end_date.strftime('%d-%m-%Y')}")
    insuff_text = (f"Ticker con dati insufficienti: {', '.join(insufficient_data)}"
                   if insufficient_data else "")

    return fig, date_values, insuff_text


# ─────────────────────────────────────────────────────────────────────────────
# Callback: sessione — toggle pannello
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('session-panel', 'style'),
    Input('session-toggle-btn', 'n_clicks'),
    State('session-panel', 'style'),
    prevent_initial_call=True,
)
def toggle_session_panel(n, current_style):
    if current_style and current_style.get('display') == 'none':
        return {'display': 'block', 'position': 'absolute', 'top': '70px', 'right': '10px',
                'z-index': '1000', 'background': 'white', 'border': '1px solid #ccc',
                'border-radius': '8px', 'box-shadow': '0 4px 20px rgba(0,0,0,0.15)',
                'padding': '16px 20px', 'width': '720px', 'max-width': '95vw'}
    return {'display': 'none'}


# ─────────────────────────────────────────────────────────────────────────────
# Callback: sessione — aggiorna lista
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('session-list-container', 'children'),
    Input('session-refresh-btn',    'n_clicks'),
    Input('session-save-btn',       'n_clicks'),
    Input('session-delete-trigger', 'data'),
    Input('session-toggle-btn',     'n_clicks'),
    prevent_initial_call=False,
)
def refresh_session_list(*_):
    sessions = list_sessions()
    if not sessions:
        return html.Div('Nessuna sessione salvata.',
                        style={'font-size': '11px', 'color': '#aaa',
                               'padding': '10px', 'text-align': 'center'})
    return [_build_session_row(r) for r in sessions]


# ─────────────────────────────────────────────────────────────────────────────
# Callback: sessione — salva
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('session-save-status', 'children'),
    Output('session-save-status', 'style'),
    Input('session-save-btn', 'n_clicks'),
    State('session-name-input', 'value'),
    State('session-desc-input', 'value'),
    *[State(sid, 'data') for sid in CLIENT_SESSION_STORES],
    prevent_initial_call=True,
)
def save_session_cb(n_clicks, name, desc, *store_values):
    if not n_clicks:
        raise PreventUpdate
    store_data = {sid: val for sid, val in zip(CLIENT_SESSION_STORES, store_values)
                  if val is not None}
    if not store_data:
        return ('⚠ Nessun dato da salvare. Carica prima i dati.',
                {'color': '#e65100', 'font-size': '10px', 'margin-top': '5px'})
    rec = save_session(name=name or '', description=desc or '', store_data=store_data)
    return (f"✅ Salvata: \"{rec['name']}\" ({rec['size_kb']} KB)",
            {'color': '#1b7a34', 'font-size': '10px', 'margin-top': '5px'})


# ─────────────────────────────────────────────────────────────────────────────
# Callback: sessione — elimina
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('session-delete-trigger', 'data'),
    Input({'type': 'session-delete-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def delete_session_cb(all_clicks):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered = ctx.triggered[0]
    if not triggered['value']:
        raise PreventUpdate
    try:
        id_dict    = json.loads(triggered['prop_id'].split('.')[0])
        session_id = id_dict['index']
    except Exception:
        raise PreventUpdate
    delete_session(session_id)
    return str(session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: sessione — seleziona per il caricamento
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('session-selected-id',  'data'),
    Output('session-load-trigger', 'data'),
    Input({'type': 'session-load-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def select_session(all_clicks):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered = ctx.triggered[0]
    if not triggered['value']:
        raise PreventUpdate
    try:
        id_dict    = json.loads(triggered['prop_id'].split('.')[0])
        session_id = id_dict['index']
    except Exception:
        raise PreventUpdate
    return session_id, session_id


@app.callback(
    *[Output(sid, 'data', allow_duplicate=True) for sid in CLIENT_SESSION_STORES],
    Input('session-load-trigger', 'data'),
    prevent_initial_call=True,
)
def load_session_cb(session_id):
    if not session_id:
        raise PreventUpdate
    store_data = load_session(session_id)
    return tuple(store_data.get(sid, no_update) for sid in CLIENT_SESSION_STORES)


# ─────────────────────────────────────────────────────────────────────────────
server = app.server   # esposto per gunicorn

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8051))
    app.run(debug=False, port=port, host='0.0.0.0')

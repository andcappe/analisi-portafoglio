"""Funzioni di calcolo delle metriche di portafoglio.

Separate da app.py così da poter essere importate (e testate) senza
avviare l'app Dash, il download dei dati o lo scheduler.
"""
import numpy as np


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

def calculate_historical_cvar(returns_series, window, tail_pct=0.05):
    # Rendimenti composti rolling a N giorni
    n_day_ret = (1 + returns_series).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1
    # CVaR calcolato sulla distribuzione espansa di tutti i rendimenti N-giorni visti fino a t
    def _cvar(w):
        n_tail = max(1, int(len(w) * tail_pct))
        return np.partition(w, n_tail)[:n_tail].mean()
    return n_day_ret.expanding(min_periods=window + 1).apply(_cvar, raw=True)

def _rolling_volatility(returns_series, window):
    return returns_series.rolling(window, min_periods=window // 2).std() * np.sqrt(252)

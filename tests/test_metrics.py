"""Test delle funzioni metriche in metrics.py.

Ogni test confronta il risultato con un calcolo di riferimento scritto
in modo indipendente (loop esplicito o formula chiusa) su dati sintetici.
"""
import numpy as np
import pandas as pd
import pytest

from metrics import (
    calculate_drawdown,
    calculate_historical_cvar,
    calculate_rolling_information_ratio,
    calculate_rolling_sharpe_ratio,
    calculate_tracking_error_volatility,
    _rolling_volatility,
)

ANN = np.sqrt(252)


@pytest.fixture
def idx():
    return pd.bdate_range('2024-01-01', periods=120)


@pytest.fixture
def asset(idx):
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx, name='A')


@pytest.fixture
def bench(idx):
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0003, 0.008, len(idx)), index=idx, name='B')


def _ref_ratio(x, window):
    """Media/deviazione standard campionaria su finestra mobile, annualizzata."""
    out = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        w = np.asarray(x[i - window + 1:i + 1])
        out[i] = w.mean() / w.std(ddof=1) * ANN
    return out


# ── Sharpe ────────────────────────────────────────────────────────────────────
class TestSharpe:
    def test_matches_reference(self, asset):
        res = calculate_rolling_sharpe_ratio(asset, window=20)
        np.testing.assert_allclose(res.to_numpy(), _ref_ratio(asset.to_numpy(), 20))

    def test_first_window_minus_one_are_nan(self, asset):
        res = calculate_rolling_sharpe_ratio(asset, window=20)
        assert res.iloc[:19].isna().all()
        assert res.iloc[19:].notna().all()

    def test_preserves_index(self, asset):
        res = calculate_rolling_sharpe_ratio(asset, window=20)
        assert res.index.equals(asset.index)

    def test_sign_follows_mean(self, idx):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.001, len(idx))
        up = pd.Series(0.01 + noise, index=idx)
        down = pd.Series(-0.01 + noise, index=idx)
        assert (calculate_rolling_sharpe_ratio(up, 10).dropna() > 0).all()
        assert (calculate_rolling_sharpe_ratio(down, 10).dropna() < 0).all()

    def test_scale_invariant(self, asset):
        a = calculate_rolling_sharpe_ratio(asset, 20)
        b = calculate_rolling_sharpe_ratio(asset * 3, 20)
        pd.testing.assert_series_equal(a, b)


# ── Information Ratio / AKRatio ──────────────────────────────────────────────
class TestInformationRatio:
    def test_matches_reference(self, asset, bench):
        res = calculate_rolling_information_ratio(asset, bench, window=30)
        active = asset.to_numpy() - bench.to_numpy()
        np.testing.assert_allclose(res.to_numpy(), _ref_ratio(active, 30))

    def test_equals_sharpe_of_active_return(self, asset, bench):
        ir = calculate_rolling_information_ratio(asset, bench, 30)
        sharpe = calculate_rolling_sharpe_ratio(asset - bench, 30)
        np.testing.assert_allclose(ir.to_numpy(), sharpe.to_numpy())

    def test_asset_vs_itself_is_nan(self, asset):
        # rendimento attivo nullo -> 0/0
        res = calculate_rolling_information_ratio(asset, asset.copy(), 20)
        assert res.isna().all()

    def test_outperformer_positive(self, asset):
        rng = np.random.default_rng(1)
        better = asset + 0.002 + rng.normal(0, 0.0005, len(asset))
        res = calculate_rolling_information_ratio(better, asset, 20).dropna()
        assert (res > 0).all()

    def test_benchmark_aligned_by_position(self, asset, bench):
        # Il benchmark viene convertito in array: conta la posizione, non l'indice.
        shifted = bench.copy()
        shifted.index = shifted.index + pd.Timedelta(days=1000)
        a = calculate_rolling_information_ratio(asset, bench, 20)
        b = calculate_rolling_information_ratio(asset, shifted, 20)
        pd.testing.assert_series_equal(a, b)
        assert a.index.equals(asset.index)

    def test_akratio_moving_average(self, asset, bench):
        # In app.py l'AKRatio è l'IR rolling, eventualmente lisciato con una media mobile.
        ir = calculate_rolling_information_ratio(asset, bench, 30)
        ak = ir.rolling(5, min_periods=1).mean()
        assert ak.iloc[29] == pytest.approx(ir.iloc[29])
        assert ak.iloc[40] == pytest.approx(ir.iloc[36:41].mean())


# ── Tracking Error Volatility ────────────────────────────────────────────────
class TestTEV:
    def test_matches_reference(self, asset, bench):
        res = calculate_tracking_error_volatility(asset, bench, window=25)
        active = (asset - bench).to_numpy()
        expected = np.full(len(active), np.nan)
        for i in range(24, len(active)):
            expected[i] = active[i - 24:i + 1].std(ddof=1) * ANN
        np.testing.assert_allclose(res.to_numpy(), expected)

    def test_zero_when_identical(self, asset):
        res = calculate_tracking_error_volatility(asset, asset.copy(), 10).dropna()
        np.testing.assert_allclose(res.to_numpy(), 0.0, atol=1e-15)

    def test_constant_spread_gives_zero(self, asset):
        res = calculate_tracking_error_volatility(asset + 0.001, asset, 10).dropna()
        np.testing.assert_allclose(res.to_numpy(), 0.0, atol=1e-12)

    def test_non_negative(self, asset, bench):
        assert (calculate_tracking_error_volatility(asset, bench, 15).dropna() >= 0).all()


# ── Drawdown ──────────────────────────────────────────────────────────────────
class TestDrawdown:
    def test_known_path(self):
        r = pd.Series([0.10, -0.20, 0.05, 0.25, -0.10])
        # valori: 1.1, 0.88, 0.924, 1.155, 1.0395 ; massimi: 1.1, 1.1, 1.1, 1.155, 1.155
        expected = [0.0, 0.88 / 1.1 - 1, 0.924 / 1.1 - 1, 0.0, 1.0395 / 1.155 - 1]
        np.testing.assert_allclose(calculate_drawdown(r).to_numpy(), expected)

    def test_monotonic_gains_never_draw_down(self):
        r = pd.Series([0.01, 0.02, 0.0, 0.03])
        assert (calculate_drawdown(r) == 0).all()

    def test_bounds(self, asset):
        dd = calculate_drawdown(asset)
        assert (dd <= 0).all()
        assert (dd >= -1).all()

    def test_total_loss(self):
        dd = calculate_drawdown(pd.Series([0.1, -1.0]))
        assert dd.iloc[-1] == pytest.approx(-1.0)

    def test_first_value_zero(self, asset):
        assert calculate_drawdown(asset).iloc[0] == 0


# ── Volatilità ────────────────────────────────────────────────────────────────
class TestVolatility:
    def test_matches_reference(self, asset):
        res = _rolling_volatility(asset, 20)
        x = asset.to_numpy()
        expected = np.full(len(x), np.nan)
        for i in range(len(x)):
            w = x[max(0, i - 19):i + 1]
            if len(w) >= 10:  # min_periods = window // 2
                expected[i] = w.std(ddof=1) * ANN
        np.testing.assert_allclose(res.to_numpy(), expected)

    def test_min_periods_is_half_window(self, asset):
        res = _rolling_volatility(asset, 20)
        assert res.iloc[:9].isna().all()
        assert res.iloc[9:].notna().all()

    def test_constant_returns_zero_vol(self, idx):
        res = _rolling_volatility(pd.Series(0.001, index=idx), 20).dropna()
        np.testing.assert_allclose(res.to_numpy(), 0.0, atol=1e-15)

    def test_scales_linearly(self, asset):
        a = _rolling_volatility(asset, 20)
        b = _rolling_volatility(asset * 2, 20)
        np.testing.assert_allclose(b.to_numpy(), 2 * a.to_numpy())


# ── CVaR storico ──────────────────────────────────────────────────────────────
def _ref_cvar(r, window, tail_pct):
    """Riferimento che replica la logica attuale di calculate_historical_cvar.

    Nota: la finestra espansa include i window-1 NaN iniziali dei rendimenti
    composti, quindi la dimensione della coda è int(t+1 * tail_pct) e non
    int(osservazioni_valide * tail_pct).
    """
    x = np.asarray(r)
    n = len(x)
    n_day = np.full(n, np.nan)
    for i in range(window - 1, n):
        n_day[i] = np.prod(1 + x[i - window + 1:i + 1]) - 1
    out = np.full(n, np.nan)
    for t in range(n):
        seen = n_day[:t + 1]
        valid = np.sort(seen[~np.isnan(seen)])
        if len(valid) < window + 1:
            continue
        n_tail = max(1, int(len(seen) * tail_pct))
        out[t] = valid[:n_tail].mean()
    return out


class TestCVaR:
    @pytest.mark.parametrize('window,tail', [(5, 0.05), (5, 0.10), (10, 0.05)])
    def test_matches_reference(self, asset, window, tail):
        res = calculate_historical_cvar(asset, window, tail)
        np.testing.assert_allclose(res.to_numpy(), _ref_cvar(asset, window, tail))

    def test_first_valid_after_window_plus_one_returns(self, asset):
        # servono window+1 rendimenti composti validi: il primo è all'indice window-1
        res = calculate_historical_cvar(asset, 5, 0.05)
        assert res.iloc[:9].isna().all()
        assert res.iloc[9:].notna().all()

    def test_equals_worst_return_when_tail_is_one(self, idx):
        # con tail_pct piccolo la coda è un solo elemento: il peggior rendimento a N giorni
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(0, 0.01, 30), index=idx[:30])
        res = calculate_historical_cvar(r, 1, 0.01)
        expected = r.expanding(min_periods=2).min()
        np.testing.assert_allclose(res.to_numpy(), expected.to_numpy())

    def test_cvar_10_less_extreme_than_5(self, asset):
        c5 = calculate_historical_cvar(asset, 5, 0.05)
        c10 = calculate_historical_cvar(asset, 5, 0.10)
        both = pd.concat([c5, c10], axis=1).dropna()
        assert (both.iloc[:, 0] <= both.iloc[:, 1] + 1e-15).all()

    def test_cvar_not_above_var_quantile(self, asset):
        res = calculate_historical_cvar(asset, 5, 0.05).dropna()
        n_day = (1 + asset).rolling(5).apply(np.prod, raw=True) - 1
        # il CVaR è una media della coda: non può superare il massimo rendimento visto
        for t in res.index:
            assert res[t] <= n_day.loc[:t].max()

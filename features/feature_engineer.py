# features/feature_engineer.py

import pandas as pd
import pandas_ta as ta
import numpy as np


def get_fractional_weights(d: float, size: int) -> np.ndarray:
    """Generates expansion weights for fractional differentiation."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] / k * (d - k + 1))
    return np.array(w[::-1])


def frac_diff_fixed_thres(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """
    Applies fractional differentiation with a fixed weight threshold.
    Preserves memory while achieving stationarity.
    """
    weights = get_fractional_weights(d, len(series))
    abs_weights = np.abs(weights)
    cum_weights = np.cumsum(abs_weights)
    if cum_weights[-1] > 0:
        cum_weights /= cum_weights[-1]

    # Drop weights below significance threshold
    lag_cutoff = np.searchsorted(cum_weights, threshold)
    weights = weights[lag_cutoff:]

    res = np.full(len(series), np.nan)
    vals = series.values

    n_weights = len(weights)
    for i in range(n_weights, len(series)):
        res[i] = np.dot(weights, vals[i - n_weights:i])

    return pd.Series(res, index=series.index, name=f"{series.name}_frac_{d}")


class TechnicalFeatureEngineer:
    def __init__(self, forward_bars: int = 5, threshold_pct: float = 0.003, frac_d: float = 0.35):
        self.forward_bars = forward_bars
        self.threshold_pct = threshold_pct
        self.frac_d = frac_d

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # -------------------------------------------------------------
        # 1. Fractional Differentiation (Stationary Price Memory)
        # -------------------------------------------------------------
        if 'close' in df.columns and len(df) > 50:
            df['close_frac_diff'] = frac_diff_fixed_thres(df['close'], d=self.frac_d)

        # -------------------------------------------------------------
        # 2. Normalized Trend Features (Stationary Distance Ratios)
        # -------------------------------------------------------------
        ema_9 = ta.ema(df['close'], length=9)
        ema_21 = ta.ema(df['close'], length=21)
        ema_50 = ta.ema(df['close'], length=50)
        ema_200 = ta.ema(df['close'], length=200)

        df['dist_ema_9'] = (df['close'] - ema_9) / df['close']
        df['dist_ema_21'] = (df['close'] - ema_21) / df['close']
        df['dist_ema_50'] = (df['close'] - ema_50) / df['close']
        df['dist_ema_200'] = (df['close'] - ema_200) / df['close']
        df['ema_spread_9_21'] = (ema_9 - ema_21) / df['close']

        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None and 'ADX_14' in adx.columns:
            df['adx'] = adx['ADX_14'] / 100.0

        st = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3.0)
        if st is not None and 'SUPERT_7_3.0' in st.columns:
            df['supertrend_dist'] = (df['close'] - st['SUPERT_7_3.0']) / df['close']
            df['supertrend_dir'] = st['SUPERTd_7_3.0']

        psar = ta.psar(df['high'], df['low'], df['close'])
        if psar is not None:
            psar_val = psar['PSARl_0.02_0.2'].fillna(psar['PSARs_0.02_0.2'])
            df['psar_dist'] = (df['close'] - psar_val) / df['close']

        # -------------------------------------------------------------
        # 3. Momentum & Oscillators (Scaled to [0, 1] or Stationarity)
        # -------------------------------------------------------------
        df['rsi'] = ta.rsi(df['close'], length=14) / 100.0

        macd = ta.macd(df['close'])
        if macd is not None and 'MACD_12_26_9' in macd.columns:
            df['macd_norm'] = macd['MACD_12_26_9'] / df['close']
            df['macd_signal_norm'] = macd['MACDs_12_26_9'] / df['close']
            df['macd_hist_norm'] = macd['MACDh_12_26_9'] / df['close']

        stoch_rsi = ta.stochrsi(df['close'])
        if stoch_rsi is not None and 'STOCHRSIk_14_14_3_3' in stoch_rsi.columns:
            df['stoch_rsi_k'] = stoch_rsi['STOCHRSIk_14_14_3_3'] / 100.0
            df['stoch_rsi_d'] = stoch_rsi['STOCHRSId_14_14_3_3'] / 100.0

        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14) / 200.0
        df['roc'] = ta.roc(df['close'], length=12) / 100.0
        df['willr'] = (ta.willr(df['high'], df['low'], df['close'], length=14) + 100.0) / 100.0

        # -------------------------------------------------------------
        # 4. Volatility & Channel Positions
        # -------------------------------------------------------------
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_norm'] = atr / df['close'] if atr is not None else np.nan

        # Parkinson Volatility
        log_hl = np.log(df['high'] / df['low'])
        df['parkinson_vol'] = np.sqrt((log_hl ** 2) / (4 * np.log(2)))

        bb = ta.bbands(df['close'], length=20, std=2.0)
        if bb is not None and 'BBP_20_2.0' in bb.columns:
            df['bb_percent'] = bb['BBP_20_2.0']
            df['bb_width'] = bb['BBB_20_2.0'] / 100.0

        kc = ta.kc(df['high'], df['low'], df['close'], length=20)
        if kc is not None and 'KCUe_20_2' in kc.columns:
            kc_range = (kc['KCUe_20_2'] - kc['KCLe_20_2']).replace(0, np.nan)
            df['kc_pos'] = (df['close'] - kc['KCLe_20_2']) / kc_range

        # -------------------------------------------------------------
        # 5. Market Structure Position
        # -------------------------------------------------------------
        donchian = ta.donchian(df['high'], df['low'], length=20)
        if donchian is not None and 'DCU_20_20' in donchian.columns:
            dc_range = (donchian['DCU_20_20'] - donchian['DCL_20_20']).replace(0, np.nan)
            df['donchian_pos'] = (df['close'] - donchian['DCL_20_20']) / dc_range

        # -------------------------------------------------------------
        # 6. Micro-Velocity & Volume Force
        # -------------------------------------------------------------
        df['velocity_3m'] = df['close'].pct_change(3)
        df['velocity_5m'] = df['close'].pct_change(5)
        df['velocity_15m'] = df['close'].pct_change(15)

        if 'volume' in df.columns:
            vol_ma = ta.sma(df['volume'], length=20)
            rel_vol = df['volume'] / vol_ma.replace(0, np.nan)
            df['volume_force'] = df['close'].pct_change(1) * rel_vol

        return df

    def create_categorical_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['future_return'] = (df['close'].shift(-self.forward_bars) - df['close']) / df['close']

        conditions = [
            (df['future_return'] > self.threshold_pct),
            (df['future_return'] < -self.threshold_pct)
        ]
        df['target'] = np.select(conditions, [1, -1], default=0)
        return df


# Alias for test suite compatibility
FeatureEngineer = TechnicalFeatureEngineer

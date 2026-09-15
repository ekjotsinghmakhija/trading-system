# eval/factor_discovery.py

import pandas as pd
import numpy as np
import pandas_ta as ta


class TripleBarrierLabeler:
    """
    Applies Lopez de Prado's Triple-Barrier Method using ATR-scaled dynamic barriers.
    Labels trades as:
        +1: Hit Upper Profit Target First (Long Edge)
        -1: Hit Lower Stop Loss First (Short Edge)
         0: Vertical Barrier Hit (Time Expired / Consolidation)
    """
    def __init__(
        self,
        pt_multiplier: float = 1.5,
        sl_multiplier: float = 1.0,
        max_holding_bars: int = 15,
        atr_length: int = 14
    ):
        self.pt_mult = pt_multiplier
        self.sl_mult = sl_multiplier
        self.max_holding = max_holding_bars
        self.atr_length = atr_length

    def compute_barriers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes dynamic upper, lower, and vertical barriers for each bar.
        """
        df = df.copy()

        # Compute ATR for dynamic volatility-scaled barrier width
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.atr_length)
        df['volatility_barrier'] = atr.fillna(df['close'] * 0.001)

        labels = np.zeros(len(df), dtype=int)
        ret_targets = np.zeros(len(df), dtype=float)
        exit_bars = np.zeros(len(df), dtype=int)

        close_vals = df['close'].values
        high_vals = df['high'].values
        low_vals = df['low'].values
        vol_vals = df['volatility_barrier'].values
        n = len(df)

        for i in range(n - self.max_holding):
            price_entry = close_vals[i]
            vol = vol_vals[i]

            pt_price = price_entry + (vol * self.pt_mult)
            sl_price = price_entry - (vol * self.sl_mult)

            hit_label = 0
            ret = 0.0
            actual_holding = self.max_holding

            # Walk forward through the vertical barrier window
            for t in range(1, self.max_holding + 1):
                curr_high = high_vals[i + t]
                curr_low = low_vals[i + t]

                # Check upper barrier hit (Profit Target)
                if curr_high >= pt_price:
                    hit_label = 1
                    ret = (pt_price - price_entry) / price_entry
                    actual_holding = t
                    break

                # Check lower barrier hit (Stop Loss)
                if curr_low <= sl_price:
                    hit_label = -1
                    ret = (sl_price - price_entry) / price_entry
                    actual_holding = t
                    break

            # If vertical barrier reached without hitting PT/SL
            if hit_label == 0:
                price_exit = close_vals[i + self.max_holding]
                ret = (price_exit - price_entry) / price_entry

            labels[i] = hit_label
            ret_targets[i] = ret
            exit_bars[i] = actual_holding

        df['tb_label'] = labels
        df['tb_return'] = ret_targets
        df['holding_period'] = exit_bars

        return df


def compute_sample_weights(df: pd.DataFrame) -> pd.Series:
    """
    Computes sample weights based on label overlap to prevent over-representing
    clustered trades in backtesting and training.
    """
    overlaps = np.zeros(len(df))
    holding = df['holding_period'].values
    n = len(df)

    for i in range(n):
        h = holding[i]
        if h > 0:
            overlaps[i: min(i + h, n)] += 1.0

    overlaps = np.where(overlaps == 0, 1.0, overlaps)
    weights = 1.0 / overlaps
    return pd.Series(weights, index=df.index, name="sample_weight")

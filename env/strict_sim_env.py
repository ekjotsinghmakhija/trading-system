import numpy as np
from scipy.stats import norm

class BSMSyntheticOptionEngine:
    """
    Calculates synthetic options prices and Greeks from futures price,
    strike, time to expiry, and implied volatility using Black-Scholes.
    """
    def __init__(self, risk_free_rate: float = 0.065):
        self.r = risk_free_rate

    def calculate_greeks(self, S: float, K: float, T: float, sigma: float, option_type: str = 'CE'):
        """
        S: Underlying Futures Price
        K: Option Strike Price
        T: Time to Expiry (in years)
        sigma: Implied Volatility (annualized, e.g., 0.15 for 15%)
        """
        if T <= 1e-6:
            # Handle expiry moment
            intrinsic = max(0.0, S - K) if option_type == 'CE' else max(0.0, K - S)
            delta = 1.0 if intrinsic > 0 else 0.0
            return {'price': intrinsic, 'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'CE':
            price = S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else: # PE
            price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1.0

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - self.r * K * np.exp(-self.r * T) * norm.cdf(d2 if option_type == 'CE' else -d2)) / 365.0
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0  # 1% change in IV

        return {
            'price': float(price),
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta),
            'vega': float(vega)
        }

class ZerodhaFrictionSimulator:
    """
    Calculates exact brokerage and statutory charges for Zerodha options trading.
    """
    def __init__(self, brokerage_per_order: float = 20.0, slippage_ticks: float = 1.0):
        self.brokerage = brokerage_per_order
        self.tick_size = 0.05
        self.slippage = slippage_ticks * self.tick_size

    def calculate_cost(self, buy_premium: float, sell_premium: float, qty: int) -> dict:
        buy_val = (buy_premium + self.slippage) * qty
        sell_val = max(0.0, (sell_premium - self.slippage)) * qty
        turnover = buy_val + sell_val

        # Statutory Charges (2026 Revised Rates)
        brokerage_total = self.brokerage * 2  # Buy + Sell
        stt = 0.0015 * sell_val  # 0.15% STT on option sell side premium
        txn_charges = 0.0003503 * turnover  # NSE Exchange Turnover Charge (~0.035%)
        sebi_charges = 0.000001 * turnover  # ₹10 per crore
        stamp_duty = 0.00003 * buy_val  # 0.003% on Buy side
        gst = 0.18 * (brokerage_total + txn_charges + sebi_charges)

        total_friction = brokerage_total + stt + txn_charges + sebi_charges + stamp_duty + gst

        return {
            'total_friction': total_friction,
            'slippage_loss': (self.slippage * 2) * qty,
            'net_pnl_drag': total_friction + ((self.slippage * 2) * qty)
        }

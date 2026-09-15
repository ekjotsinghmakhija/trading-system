# execution/order_router.py

import time
import logging
from typing import Dict, Any

logger = logging.getLogger("OrderRouter")

class MockPaperBroker:
    """
    Simulates live options execution against level-2 orderbook depth feeds.
    """
    def __init__(self, latency_ms: int = 250, slippage_ticks: float = 1.0):
        self.latency_sec = latency_ms / 1000.0
        self.tick_size = 0.05
        self.slippage = slippage_ticks * self.tick_size

    def execute_order(self, symbol: str, transaction_type: str, price: float, quantity: int) -> Dict[str, Any]:
        # Inject simulated network latency
        time.sleep(self.latency_sec)

        # Apply slippage (Buy higher than ask, Sell lower than bid)
        if transaction_type.upper() == 'BUY':
            fill_price = price + self.slippage
        else:
            fill_price = max(0.05, price - self.slippage)

        fill_price = round(fill_price / self.tick_size) * self.tick_size

        return {
            'status': 'COMPLETE',
            'symbol': symbol,
            'transaction_type': transaction_type,
            'requested_price': price,
            'filled_price': fill_price,
            'quantity': quantity,
            'slippage_paid': self.slippage * quantity,
            'timestamp': time.time()
        }


class OrderRouter:
    """
    Routes execution signals to either Live Zerodha Kite API or Mock Paper Engine.
    """
    def __init__(self, paper_trading: bool = True, kite_client=None):
        self.paper_trading = paper_trading
        self.kite = kite_client
        self.mock_broker = MockPaperBroker()

    def route_signal(
        self,
        futures_price: float,
        signal: int,
        volatility: float,
        quantity: int = 50
    ) -> Dict[str, Any]:
        """
        Maps a futures signal (+1 Long, -1 Short) to an ATM options order.
        """
        if signal == 0:
            return {'status': 'SKIPPED', 'reason': 'NEUTRAL_SIGNAL'}

        # Strike mapping: Round futures price to nearest 50-point step
        atm_strike = int(round(futures_price / 50.0) * 50)
        option_type = 'CE' if signal == 1 else 'PE'
        symbol = f"NIFTY26SEP{atm_strike}{option_type}"

        # Estimate option premium baseline from futures distance
        estimated_premium = max(10.0, volatility * 50.0)

        if self.paper_trading:
            logger.info(f"[PAPER ORDER] {signal} | Futures: {futures_price} | Strike: {atm_strike}{option_type}")
            return self.mock_broker.execute_order(
                symbol=symbol,
                transaction_type='BUY',
                price=estimated_premium,
                quantity=quantity
            )
        else:
            if self.kite is None:
                raise ValueError("Live Kite API client not initialized.")
            # Live Zerodha execution call
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NFO,
                tradingsymbol=symbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                quantity=quantity,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_MARKET
            )
            return {'status': 'SUBMITTED', 'order_id': order_id}

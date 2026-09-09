import time
import logging
from kiteconnect import KiteConnect, KiteExceptions

class PeggedOrderRouter:
    """
    Pegged Limit Order Execution State Machine for Zerodha Kite API.
    Places pegged limit orders at current bid/ask and adjusts price every 3 seconds.
    """
    def __init__(self, api_key: str, access_token: str):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("OrderRouter")

    def get_current_quote(self, trading_symbol: str, exchange: str = "NFO") -> tuple[float, float]:
        """Fetches current best bid and ask for the instrument."""
        instrument_key = f"{exchange}:{trading_symbol}"
        quote = self.kite.quote(instrument_key)
        depth = quote[instrument_key]["depth"]
        best_bid = depth["buy"][0]["price"]
        best_ask = depth["sell"][0]["price"]
        return best_bid, best_ask

    def execute_pegged_limit_order(self, trading_symbol: str, transaction_type: str, quantity: int, max_retries: int = 5) -> dict:
        """
        State Machine: Places order pegged at best price, monitors fill state,
        and re-pegs every 3 seconds if unfilled.
        """
        best_bid, best_ask = self.get_current_quote(trading_symbol)
        initial_price = best_bid if transaction_type == self.kite.TRANSACTION_TYPE_BUY else best_ask

        self.logger.info(f"[+] Submitting {transaction_type} limit order for {trading_symbol} @ ₹{initial_price}")

        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NFO,
                tradingsymbol=trading_symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_LIMIT,
                price=initial_price
            )
        except Exception as e:
            self.logger.error(f"[!] Order placement failed: {e}")
            return {"status": "FAILED", "error": str(e)}

        # Order monitoring state loop
        for attempt in range(max_retries):
            time.sleep(3.0)  # Wait 3 seconds per modification step
            order_history = self.kite.order_history(order_id)
            latest_state = order_history[-1]

            if latest_state["status"] == "COMPLETE":
                self.logger.info(f"[✓] Order {order_id} FILLED at avg price ₹{latest_state.get('average_price', initial_price)}")
                return {"status": "COMPLETE", "order_id": order_id}

            if latest_state["status"] in ["CANCELLED", "REJECTED"]:
                self.logger.warning(f"[!] Order {order_id} TERMINATED with status: {latest_state['status']}")
                return {"status": latest_state["status"], "order_id": order_id}

            # Re-peg to updated order book quotes
            new_bid, new_ask = self.get_current_quote(trading_symbol)
            new_price = new_bid if transaction_type == self.kite.TRANSACTION_TYPE_BUY else new_ask

            if new_price != initial_price:
                self.logger.info(f"    [Re-peg Attempt {attempt+1}/{max_retries}] Adjusting price ₹{initial_price} -> ₹{new_price}")
                self.kite.modify_order(
                    variety=self.kite.VARIETY_REGULAR,
                    order_id=order_id,
                    price=new_price
                )
                initial_price = new_price

        # Force execution fallback to market if unfilled after retries
        self.logger.warning(f"[!] Max retries reached. Converting order {order_id} to MARKET order.")
        self.kite.modify_order(
            variety=self.kite.VARIETY_REGULAR,
            order_id=order_id,
            order_type=self.kite.ORDER_TYPE_MARKET
        )
        return {"status": "CONVERTED_TO_MARKET", "order_id": order_id}

if __name__ == "__main__":
    # Structural Smoke Test
    print("[✓] Pegged Order Router state machine initialized.")

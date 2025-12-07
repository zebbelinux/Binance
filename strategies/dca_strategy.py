
from typing import Dict, List, Any
from datetime import datetime
import logging
from strategies.base_strategy import BaseStrategy

class DCAStrategy(BaseStrategy):
    """
    Crash regime safety strategy: either wait or accumulate small DCA buys.
    Config:
      dca_step_pct: 0.05   # every -5% from anchor add small buy
      dca_max_steps: 6
      dca_budget_fraction: 0.10  # total budget portion of portfolio
      per_step_fraction: 0.015   # per step fraction of portfolio
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DCA", config)
        self.dca_step_pct = config.get('dca_step_pct', 0.05)
        self.dca_max_steps = config.get('dca_max_steps', 6)
        self.dca_budget_fraction = config.get('dca_budget_fraction', 0.10)
        self.per_step_fraction = config.get('per_step_fraction', 0.015)
        self.anchor_price = {}   # symbol -> anchor
        self.steps_done = {}     # symbol -> steps used

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        try:
            symbol = market_data.get('symbol', 'BTCTRY')
            price = float(market_data.get('price', 0.0) or 0.0)
            if price <= 0:
                return signals

            if symbol not in self.anchor_price:
                # set first anchor when strategy starts
                self.anchor_price[symbol] = price
                self.steps_done[symbol] = 0
                return signals  # wait until first step distance occurs

            anchor = self.anchor_price[symbol]
            steps = self.steps_done.get(symbol, 0)
            if steps >= self.dca_max_steps:
                return signals

            # next target price to buy
            target = anchor * (1 - self.dca_step_pct * (steps + 1))
            if price <= target:
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'strength': 0.5,
                    'entry_price': price,
                    'stop_loss': None,
                    'take_profit': None,
                    'reason': f'DCA step {steps+1}/{self.dca_max_steps} at {price:.2f}',
                    'strategy': 'dca'
                })
                self.steps_done[symbol] = steps + 1
            return signals
        except Exception as e:
            logging.getLogger(__name__).error(f"DCA generate_signals error: {e}")
            return signals

    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        # total budget cap and per-step fraction
        total_cap = account_balance * self.dca_budget_fraction
        per_step_cap = account_balance * self.per_step_fraction
        # ensure we don't exceed total cap based on steps used
        symbol = signal.get('symbol', 'BTCTRY')
        steps = self.steps_done.get(symbol, 0)
        if steps * per_step_cap >= total_cap:
            return 0.0
        return per_step_cap

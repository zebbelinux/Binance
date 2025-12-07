from typing import Dict, Any
from strategies.hedge_strategy import HedgeStrategy, HedgeStrategyType

class LeadLagStrategy(HedgeStrategy):
    """Lead-Lag Trading için HedgeStrategy sarmalayıcısı"""
    def __init__(self, config: Dict[str, Any] | None = None):
        cfg = dict(config or {})
        cfg['strategy_type'] = HedgeStrategyType.LEAD_LAG_TRADING.value
        super().__init__(cfg)

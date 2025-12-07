from typing import Dict, Any
from strategies.hedge_strategy import HedgeStrategy, HedgeStrategyType

class CointegrationStrategy(HedgeStrategy):
    """Cointegration/Statistical Arbitrage için HedgeStrategy sarmalayıcısı"""
    def __init__(self, config: Dict[str, Any] | None = None):
        cfg = dict(config or {})
        cfg['strategy_type'] = HedgeStrategyType.STATISTICAL_ARBITRAGE.value
        super().__init__(cfg)

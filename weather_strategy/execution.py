from __future__ import annotations

from weather_strategy.models import TradeSignal


class ExecutionAdapter:
    def place_order(self, signal: TradeSignal) -> str:
        raise NotImplementedError


class DisabledLiveExecutionAdapter(ExecutionAdapter):
    def place_order(self, signal: TradeSignal) -> str:
        raise RuntimeError(
            "Live execution is disabled. Paper trade this strategy first, then add a Polymarket API/MCP adapter here."
        )


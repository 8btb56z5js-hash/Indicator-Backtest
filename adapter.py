"""Dynamic Parameter Adapter for DCS Indicator

This module provides a simple interface to retrieve trading parameters
based on the selected trading mode. It is designed to be easily
extended or modified for additional modes or parameters.

Supported modes:
- "day":   Day‑trading configuration
- "swing": Swing‑trading configuration

Each mode defines its own set of parameters such as RSI periods,
Support/Resistance (S/R) time‑frame, and recommended exit times.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class TradingParameters:
    """Container for trading parameters for a given mode."""
    rsi_period: int
    sr_timeframe: str
    exit_time: str
    additional: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a plain dictionary for convenience."""
        base = {
            "rsi_period": self.rsi_period,
            "sr_timeframe": self.sr_timeframe,
            "exit_time": self.exit_time,
        }
        if self.additional:
            base.update(self.additional)
        return base


# Default parameter sets for the two supported modes
_DEFAULT_PARAMS: Dict[str, TradingParameters] = {
    "day": TradingParameters(
        rsi_period=14,
        sr_timeframe="5m",
        exit_time="close_of_day",
        additional={"max_positions": 5, "stop_loss": 0.02},
    ),
    "swing": TradingParameters(
        rsi_period=21,
        sr_timeframe="1h",
        exit_time="close_of_week",
        additional={"max_positions": 2, "stop_loss": 0.05},
    ),
}


def get_parameters(mode: str) -> TradingParameters:
    """Retrieve the TradingParameters for a given mode.

    Args:
        mode: Either "day" or "swing" (case‑insensitive).

    Returns:
        TradingParameters instance for the requested mode.

    Raises:
        KeyError: If an unsupported mode is supplied.
    """
    key = mode.lower()
    if key not in _DEFAULT_PARAMS:
        raise KeyError(f"Unsupported mode '{mode}'. Supported modes: {list(_DEFAULT_PARAMS)}")
    return _DEFAULT_PARAMS[key]


def list_modes() -> list:
    """Return a list of all supported trading modes."""
    return list(_DEFAULT_PARAMS.keys())


def add_mode(mode: str, params: TradingParameters) -> None:
    """Add or overwrite a trading mode configuration at runtime.

    This function mutates the internal dictionary, allowing dynamic
    extension without modifying the source file.
    """
    _DEFAULT_PARAMS[mode.lower()] = params


if __name__ == "__main__":
    # Simple demo when run directly
    import json
    for m in list_modes():
        print(f"Mode: {m}\nParameters: {json.dumps(get_parameters(m).to_dict(), indent=2)}\n")

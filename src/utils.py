from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import math

def parse_dt(x: Optional[str]) -> Optional[datetime]:
    if x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() == "":
        return None
    return datetime.fromisoformat(str(x))

def clamp(lo: float, x: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

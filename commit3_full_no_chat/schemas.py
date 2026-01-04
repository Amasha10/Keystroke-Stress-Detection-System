from pydantic import BaseModel
from typing import List, Optional


class KeystrokeEvent(BaseModel):
    key: str
    press_time: float
    release_time: float
    is_backspace: Optional[bool] = False

class KeystrokeRequest(BaseModel):
    events: List[KeystrokeEvent]


class StressResponse(BaseModel):
    stress_probability: float
    stress_pred: int

"""Stage 3: reimplement the observation builder.

Reference: domains/minirl/obs.py
"""
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ObsBuilder

from domains.minirl.state import (
    BALL_MAX_SPEED, CAR_MAX_SPEED, POS_NORM, MiniRLState, wrap_angle,
)


class DefaultObs(ObsBuilder[int, np.ndarray, MiniRLState, int]):
    def get_obs_space(self, agent: int) -> int:
        raise NotImplementedError("Return the number of dimensions in one observation.")

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError("Nothing to reset here — what should the body be?")

    def build_obs(self, agents: List[int], state: MiniRLState,
                  shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        # Return one float32 vector per agent. Normalize everything to roughly
        # [-1, 1], and encode yaw as (cos, sin) rather than a raw angle.
        raise NotImplementedError

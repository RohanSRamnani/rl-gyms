"""State types and tuning constants for ContentSim.

1 tick = 1 simulated day. One episode = one 30-day strategy cycle.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# --- Scale -------------------------------------------------------------------
N_FORMATS = 3
FORMAT_NAMES = ("slideshow", "video", "talking_head")
N_HOOKS = 4
HOOK_NAMES = ("curiosity_list", "pov", "question", "transformation")
N_TOPICS = 3
TOPIC_NAMES = ("home_styling", "slow_living", "product_story")
HISTORY_K = 3                       # posts visible to the agent
CYCLE_TICKS = 30                    # one episode = one strategy cycle

# --- Action space ------------------------------------------------------------
N_ACTIONS = N_FORMATS * N_HOOKS * N_TOPICS + 1   # 36 posts + skip = 37
SKIP_ACTION = N_ACTIONS - 1

# --- Audience model ----------------------------------------------------------
V0 = 2000.0                         # base views at F0 followers, quality 1
F0 = 5000.0                         # reference follower count
REACH_EXPONENT = 0.6                # sublinear reach growth
VIEW_NOISE_SIGMA = 0.9              # lognormal sigma — the heavy tail is load-bearing
FOLLOW_RATE = 0.01                  # follows = rate * views * min(quality, 2)
INIT_FOLLOWERS_MIN = 1_000.0
INIT_FOLLOWERS_MAX = 20_000.0

# --- Hidden dynamics ---------------------------------------------------------
AFFINITY_MIN, AFFINITY_MAX = 0.2, 2.0            # clip range for meta / topic_heat
AFFINITY_INIT_MIN, AFFINITY_INIT_MAX = 0.5, 1.5  # per-episode init range
DRIFT_STD = 0.04                    # per-day random-walk step
JUMP_PROB = 0.03                    # per-day chance a format's meta reshuffles
SIGNAL_NOISE_STD = 0.15             # obs noise on trend/heat signals

# --- Fatigue / authority (the convergence tension) ---------------------------
FATIGUE_DECAY = 0.7                 # per-day multiplier; +1 per post of a format
MAX_FATIGUE_NORM = 4.0              # obs normalizer (steady-state ~2.3 for daily)
AUTHORITY_GAIN = 0.15               # per post of a topic
AUTHORITY_DECAY = 0.9               # per day when a post was made
AUTHORITY_SKIP_DECAY = 0.8          # per day when the account skipped
AUTHORITY_MAX = 1.0
AUTHORITY_BOOST = 0.5               # views multiplier = 1 + boost * authority

# --- Flag mechanic (unoriginal-content analog) -------------------------------
REPEAT_FLAG_STREAK = 3              # identical posts in a row before rolls start
STRIKE_PROB = 0.25                  # per qualifying day
MAX_STRIKES = 3                     # strikes -> restricted -> termination

# --- Outliers ----------------------------------------------------------------
OUTLIER_MULTIPLIER = 2.0            # views >= 2x trailing median
MIN_PRIOR_POSTS = 3                 # median needs a base

# --- Normalizers / reward references -----------------------------------------
MAX_FOLLOWERS_NORM = 200_000.0
VIEWS_REF = 50_000.0
FOLLOWS_REF = 500.0


@dataclass
class PostRecord:
    format: int
    hook: int
    topic: int
    day: int
    views: float
    follows: float


@dataclass
class ContentState:
    followers: float = 5000.0
    # HIDDEN — the environment's truth; must never leak into the obs.
    meta: np.ndarray = field(default_factory=lambda: np.ones(N_FORMATS))
    topic_heat: np.ndarray = field(default_factory=lambda: np.ones(N_TOPICS))
    hook_fit: np.ndarray = field(default_factory=lambda: np.ones((N_FORMATS, N_HOOKS)))
    # OBSERVED — noisy daily readings of the hidden state (the "research corpus").
    trend_signals: np.ndarray = field(default_factory=lambda: np.ones(N_FORMATS))
    heat_signals: np.ndarray = field(default_factory=lambda: np.ones(N_TOPICS))
    # Account condition.
    fatigue: np.ndarray = field(default_factory=lambda: np.zeros(N_FORMATS))
    authority: np.ndarray = field(default_factory=lambda: np.zeros(N_TOPICS))
    flag_strikes: int = 0
    restricted: bool = False
    had_outlier: bool = False
    history: List[PostRecord] = field(default_factory=list)
    ticks: int = 0
    # Streak bookkeeping for the flag mechanic.
    last_action: Optional[Tuple[int, int, int]] = None
    repeat_streak: int = 0
    # Per-tick reward channels, cleared at the start of every engine tick.
    views_today: float = 0.0
    follows_today: float = 0.0
    outlier_today: bool = False
    posted_today: bool = False


def decode_action(action: int) -> Optional[Tuple[int, int, int]]:
    """Action int -> (format, hook, topic), or None for skip.

    THE single action authority: the engine resolves actions through this and
    the scripted baselines encode with its inverse. If they ever diverge, the
    agent posts a different package than the one it chose.
    """
    if action == SKIP_ACTION:
        return None
    format_index, rest = divmod(action, N_HOOKS * N_TOPICS)
    hook_index, topic_index = divmod(rest, N_TOPICS)
    return format_index, hook_index, topic_index

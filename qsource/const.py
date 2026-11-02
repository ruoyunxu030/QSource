"""Simulation constants. Seed layout matches the paper runs."""

from __future__ import annotations

# Market / protocol
PKT_ROWS = 40
PKTS_PER_SELLER = 5
N_SELLERS = 8
Q_LEVELS = [0.0, 0.1, 0.3, 0.5]
S1_FIXED = 5  # rows in the discarded 5-row trial offer (RNG must still run)
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_BUYERS = 30
B = 20.0
K = 3
RHO = 7.0
N_SEED = 100
FLOOR_F_MAX = 80
W_MAX = 0.25
DELTA = 0.05
SKIP_THRESHOLD = 1.0
BETA_A, BETA_B = 2.0, 1.0
FIXED_MU0 = 2.0 / 3.0
FIXED_TAU0_2 = 1.0 / 18.0
TAU0_FLOOR = 0.0025
CLIP_ROW = (1e-3, 1.0 - 1e-3)
CLIP_BUYER = (0.05, 1.0)
Z_LB = 0.8416
EXTREME_K = 2.0
ETA_DEFAULT = 0.1
ETA_ONE = 1.0 / 31.0
N_ATK = {0.1: 3, 0.2: 8, 0.3: 13}

# Seeds. POLICY_SEED_ID numbers are part of the RNG; do not renumber.
SEED_BASE = 20261019
DS_ID = {"domainnet": 1, "amazon": 2}
PHASE_TEST = 2
BUYER_SALT = 1001
INJECTION_SALT = 3002
FLOOR_SALT = 1006
PLATFORM_SALT = 1007
LOCK = {
    "amazon": dict(beta=0.35, s_rows=20, sampling_rule="value"),
    "domainnet": dict(beta=0.25, s_rows=10, sampling_rule="value"),
}

DESIGNS = (
    "no_sharing", "row_level", "buyer_level", "buyer_level_floor", "known_quality",
)
# Values are the historical RNG ids (0/1/4/9/18). Do not renumber.
POLICY_SEED_ID = {
    "no_sharing": 0,
    "row_level": 1,
    "known_quality": 4,
    "buyer_level": 9,
    "buyer_level_floor": 18,
}
ROW_DESIGNS = ("no_sharing", "row_level")
BUYER_DESIGNS = ("buyer_level", "buyer_level_floor")
FLOOR_DESIGNS = ("buyer_level_floor",)
FLOOR_ROWS = {"buyer_level_floor": 40}
CAL_PRIOR = {"buyer_level": "eb", "buyer_level_floor": "eb"}

# Honest-record 5% quantile L and π=0.5 median, from the paper calibration.
L = {
    "amazon": 0.6512159931550242,
    "domainnet": 0.9040932984060011,
}
MEDIOCRE = {
    "amazon": 0.5609767644634186,
    "domainnet": 0.9358793332387914,
}

# Median of honest π=1 records from a no-injection run of this market
# (printed to 4 d.p. as 0.9616 / 0.9808 in the paper).
RAISE_WEAK = {
    "amazon": 0.961628827221886,
    "domainnet": 0.9808480688625696,
}

KIND_PACKET = "packet"
KIND_TRIAL = "trial"
KIND_TRIAL5 = "trial5"
SRC_BACKED = "packet_backed"
SRC_PLATFORM = "platform_checked"
SRC_TRIAL = "trial_only"
BACKED_SOURCES = frozenset({SRC_BACKED, SRC_PLATFORM})
KIND_BUYER = "buyer"
KIND_ROW = "row"

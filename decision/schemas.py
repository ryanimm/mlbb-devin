"""Contracts for mlbb.state/v1 and mlbb.decision/v1.

These models define the interface between perception, the two-tier
decision layer, and the command bus. They are the blueprint's schema
only — no decision or training loop is implemented yet.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# State: mlbb.state/v1
# ---------------------------------------------------------------------------

class Cooldowns(BaseModel):
    """Remaining cooldown in seconds. 0 means the skill is ready."""

    s1: float = 0
    s2: float = 0
    ult: float = 0
    spell: float = 0


class SelfState(BaseModel):
    hero: str
    role: str
    pos: list[float] | None = None
    zone: str | None = None
    hp: float | None = Field(default=None, ge=0, le=1)
    mana: float | None = Field(default=None, ge=0, le=1)
    level: int
    cd: Cooldowns
    spell: str | None = None
    gold: int = 0
    alive: bool = True


class AllyState(BaseModel):
    hero: str
    role: str
    pos: list[float] | None = None
    zone: str | None = None
    hp: float | None = Field(default=None, ge=0, le=1)
    level: int | None = None
    ult_ready: bool | None = None
    alive: bool = True
    respawn_in: float | None = None


class VisibleEnemy(BaseModel):
    hero: str
    pos: list[float] | None = None
    zone: str | None = None
    hp: float | None = Field(default=None, ge=0, le=1)
    level: int | None = None
    last_seen: float = 0.0


class MissingEnemy(BaseModel):
    hero: str
    last_zone: str | None = None
    last_seen: float


class Objective(BaseModel):
    alive: bool
    spawn_in: float = 0
    contested: bool | None = None


class Objectives(BaseModel):
    turtle: Objective
    lord: Objective


class Towers(BaseModel):
    """Per-lane remaining towers as [T1, T2, base]; 1 = standing, 0 = destroyed."""

    ally: dict[str, list[int]]
    enemy: dict[str, list[int]]


class Lead(BaseModel):
    gold: int
    xp: int
    kills: list[int]  # [ally, enemy]


class GameState(BaseModel):
    """Compact match snapshot produced by the state builder."""

    model_config = ConfigDict(populate_by_name=True)

    schema_id: Literal["mlbb.state/v1"] = Field(default="mlbb.state/v1", alias="schema")
    t: float = Field(description="Game clock in seconds")
    seq: int = Field(description="Monotonic snapshot id")
    self: SelfState
    allies: list[AllyState] = Field(default_factory=list)
    enemies_visible: list[VisibleEnemy] = Field(default_factory=list)
    enemies_missing: list[MissingEnemy] = Field(default_factory=list)
    objectives: Objectives
    towers: Towers
    lead: Lead
    minimap_threat: dict[str, float] = Field(default_factory=dict)


def compact(state: GameState) -> dict:
    """Shrink a snapshot for the strategist prompt.

    Drops obvious defaults, rounds floats to two decimals, and prefers
    zone names over raw coordinates so a snapshot fits in a few hundred tokens.
    """

    def _round(value):
        if isinstance(value, float):
            return round(value, 2)
        if isinstance(value, list):
            return [_round(v) for v in value]
        if isinstance(value, dict):
            return {k: _round(v) for k, v in value.items() if v is not None}
        return value

    payload = _round(state.model_dump(by_alias=True, exclude_none=True))
    # Prefer zone labels; coordinates stay only when a zone is missing.
    units = []
    if isinstance(payload.get("self"), dict):
        units.append(payload["self"])
    units.extend(payload.get("allies") or [])
    units.extend(payload.get("enemies_visible") or [])
    for unit in units:
        if isinstance(unit, dict) and unit.get("zone"):
            unit.pop("pos", None)
    return payload


# ---------------------------------------------------------------------------
# Decision: mlbb.decision/v1
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    """High-level intents published on the command bus."""

    # Macro (strategist)
    FARM_LANE = "FARM_LANE"
    FARM_JUNGLE = "FARM_JUNGLE"
    ROTATE = "ROTATE"
    GANK = "GANK"
    PUSH = "PUSH"
    SPLIT_PUSH = "SPLIT_PUSH"
    DEFEND_TOWER = "DEFEND_TOWER"
    CONTEST_OBJECTIVE = "CONTEST_OBJECTIVE"
    FOCUS_LORD = "FOCUS_LORD"
    FOCUS_TURTLE = "FOCUS_TURTLE"
    GROUP = "GROUP"
    WARD_VISION = "WARD_VISION"
    RECALL = "RECALL"
    INVADE = "INVADE"
    # Micro (tactical)
    ENGAGE = "ENGAGE"
    DISENGAGE = "DISENGAGE"
    RETREAT = "RETREAT"
    KITE = "KITE"
    FOCUS_TARGET = "FOCUS_TARGET"
    PEEL = "PEEL"
    DODGE = "DODGE"
    HOLD = "HOLD"


MACRO_INTENTS = frozenset(
    {
        Intent.FARM_LANE,
        Intent.FARM_JUNGLE,
        Intent.ROTATE,
        Intent.GANK,
        Intent.PUSH,
        Intent.SPLIT_PUSH,
        Intent.DEFEND_TOWER,
        Intent.CONTEST_OBJECTIVE,
        Intent.FOCUS_LORD,
        Intent.FOCUS_TURTLE,
        Intent.GROUP,
        Intent.WARD_VISION,
        Intent.RECALL,
        Intent.INVADE,
    }
)

# Tactical may override the macro plan only with these defensive intents.
DEFENSIVE_INTENTS = frozenset({Intent.RETREAT, Intent.DISENGAGE, Intent.DODGE})


class TargetType(str, Enum):
    HERO = "hero"
    ZONE = "zone"
    OBJECTIVE = "objective"
    TOWER = "tower"


class Target(BaseModel):
    type: TargetType | str
    id: str


class NextAction(BaseModel):
    """Optional follow-up the strategist wants after a delay."""

    intent: Intent
    target: str | None = None
    after_s: float | None = None


class Decision(BaseModel):
    """High-level command published on the command bus."""

    model_config = ConfigDict(populate_by_name=True)

    schema_id: Literal["mlbb.decision/v1"] = Field(
        default="mlbb.decision/v1", alias="schema"
    )
    seq: int = Field(description="Snapshot this decision answers")
    tier: Literal["strategist", "tactical"]
    intent: Intent
    target: Target | None = None
    lane: str | None = None
    urgency: float = Field(ge=0, le=1, description="Used by execution to preempt")
    ttl_ms: int = Field(
        default=1000, description="Expired commands yield to default behavior"
    )
    confidence: float = Field(default=0.5, ge=0, le=1)
    reasoning: str = ""
    next: NextAction | None = None


# ---------------------------------------------------------------------------
# Illustrative payloads (not recorded match data)
# ---------------------------------------------------------------------------

EXAMPLE_STATE: dict = {
    "schema": "mlbb.state/v1",
    "t": 412.6,
    "seq": 8231,
    "self": {
        "hero": "Chou",
        "role": "roam",
        "pos": [0.42, 0.61],
        "zone": "river_bot",
        "hp": 0.38,
        "mana": 0.55,
        "level": 7,
        "cd": {"s1": 0, "s2": 3.2, "ult": 18.0, "spell": 0},
        "spell": "flicker",
        "gold": 3120,
        "alive": True,
    },
    "allies": [
        {
            "hero": "Layla",
            "role": "gold",
            "pos": [0.71, 0.88],
            "zone": "lane_bot",
            "hp": 0.62,
            "level": 6,
            "ult_ready": True,
            "alive": True,
        },
        {
            "hero": "Fanny",
            "role": "jungle",
            "pos": None,
            "zone": "jungle_top",
            "hp": None,
            "level": 8,
            "alive": False,
            "respawn_in": 11,
        },
    ],
    "enemies_visible": [
        {
            "hero": "Franco",
            "pos": [0.50, 0.55],
            "zone": "river_bot",
            "hp": 0.80,
            "level": 7,
            "last_seen": 0.0,
        }
    ],
    "enemies_missing": [
        {"hero": "Ling", "last_zone": "jungle_bot", "last_seen": 6.4}
    ],
    "objectives": {
        "turtle": {"alive": True, "spawn_in": 0, "contested": False},
        "lord": {"alive": False, "spawn_in": 68},
    },
    "towers": {
        "ally": {"top": [1, 1, 1], "mid": [0, 1, 1], "bot": [1, 1, 1]},
        "enemy": {"top": [1, 1, 1], "mid": [1, 1, 1], "bot": [0, 1, 1]},
    },
    "lead": {"gold": -850, "xp": -200, "kills": [6, 8]},
    "minimap_threat": {"lane_bot": 0.7, "river_bot": 0.6, "mid": 0.2},
}

EXAMPLE_DECISION: dict = {
    "schema": "mlbb.decision/v1",
    "seq": 8231,
    "tier": "strategist",
    "intent": "RETREAT",
    "target": {"type": "zone", "id": "tower_bot_t2"},
    "lane": "bot",
    "urgency": 0.85,
    "ttl_ms": 1500,
    "confidence": 0.7,
    "reasoning": (
        "HP 38%, Ling missing 6s from jungle_bot, 2 vs 3 visible. "
        "Wait for Fanny to respawn in 11s, then contest turtle."
    ),
    "next": {"intent": "CONTEST_OBJECTIVE", "target": "turtle", "after_s": 11},
}


if __name__ == "__main__":
    GameState.model_validate(EXAMPLE_STATE)
    Decision.model_validate(EXAMPLE_DECISION)
    print("schema examples ok")

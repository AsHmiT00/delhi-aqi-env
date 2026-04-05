"""
Pydantic typed models for the Delhi AQI Policy Environment.
Required by OpenEnv spec: all observations, actions, rewards must be typed.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class AQIObservation(BaseModel):
    """
    What the agent sees each timestep (every 6 hours).
    All values are in real units — not normalized.
    The LLM agent reads these directly as text.
    """
    current_aqi: float = Field(..., ge=0, le=600,
        description="Current PM2.5 AQI reading in Delhi")
    forecast_24h: float = Field(..., ge=0, le=600,
        description="Predicted AQI 24 hours from now")
    forecast_48h: float = Field(..., ge=0, le=600,
        description="Predicted AQI 48 hours from now — USE THIS for proactive action")
    wind_speed: float = Field(..., ge=0, le=30,
        description="Wind speed in m/s. Higher = more dispersion = better air")
    wind_direction: float = Field(..., ge=0, le=360,
        description="Wind direction degrees. 315° (NW) = stubble smoke incoming from Punjab")
    mixing_height: float = Field(..., ge=50, le=3000,
        description="Atmospheric mixing height in meters. Low (<500) = pollution trapped near ground")
    humidity: float = Field(..., ge=0, le=100,
        description="Relative humidity %. High humidity worsens PM2.5 particle formation")
    stubble_index: float = Field(..., ge=0, le=100,
        description="Punjab/Haryana crop fire intensity 0-100. Above 60 = major pollution incoming")
    step_number: int = Field(..., ge=0,
        description="Current episode step (each step = 6 hours)")
    day_of_week: int = Field(..., ge=0, le=6,
        description="0=Monday, 6=Sunday. Affects economic cost of restrictions")
    active_policies: List[int] = Field(...,
        description="Currently active policy levels [truck_ban, odd_even, construction, industry_cap, schools, wfh]")
    aqi_category: str = Field(...,
        description="Human-readable AQI category: Good/Moderate/Poor/Very Poor/Severe")


class PolicyAction(BaseModel):
    """
    6 policy levers the agent controls each timestep.
    Each lever has discrete intensity levels.
    """
    truck_ban: int = Field(..., ge=0, le=2,
        description="0=no ban, 1=night only (10pm-6am), 2=full day ban except essentials")
    odd_even: int = Field(..., ge=0, le=2,
        description="0=off, 1=peak hours only (8am-8pm), 2=all day")
    construction_ban: int = Field(..., ge=0, le=2,
        description="0=normal operations, 1=strict dust norms enforced, 2=non-essential construction halted")
    industry_cap: int = Field(..., ge=0, le=3,
        description="0=100% capacity, 1=75% capacity, 2=50% capacity, 3=essential only (25%)")
    schools_closed: int = Field(..., ge=0, le=1,
        description="0=schools open, 1=schools closed (reduces traffic + child exposure)")
    wfh_level: int = Field(..., ge=0, le=2,
        description="0=no advisory, 1=WFH recommended, 2=mandatory for government offices")


class AQIReward(BaseModel):
    """
    Reward signal with full breakdown.
    Total = health_component + economic_component + proactive_bonus
    """
    total: float = Field(...,
        description="Total reward for this timestep")
    health_component: float = Field(...,
        description="Positive reward for good air, heavy penalty for severe AQI")
    economic_component: float = Field(...,
        description="Negative cost from active restrictions (economic disruption)")
    proactive_bonus: float = Field(...,
        description="Bonus when agent acts early before AQI reaches crisis level")


class StepResult(BaseModel):
    """Return type of step() — full episode transition."""
    observation: AQIObservation
    reward: AQIReward
    done: bool
    info: Dict[str, Any] = {}


class EpisodeConfig(BaseModel):
    """Configuration for a specific task scenario."""
    task_id: str
    initial_aqi: float
    initial_stubble: float
    max_steps: int
    scenario_name: str
    difficulty: str

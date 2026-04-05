"""
Delhi AQI Policy Environment — OpenEnv Compatible

The core environment class. This is what judges evaluate.
Implements the OpenEnv interface:
  - reset(task) → AQIObservation
  - step(PolicyAction) → StepResult
  - state() → AQIObservation

No external data files needed — synthetic weather generated from
real Delhi statistical distributions.
"""

import numpy as np
from typing import Optional, Dict, Any

from models import (
    AQIObservation,
    PolicyAction,
    AQIReward,
    StepResult,
    EpisodeConfig,
)
from dispersion_model import DispersionModel, generate_episode_weather


# ─── Task scenario definitions ───────────────────────────────────────────────

TASK_CONFIGS: Dict[str, EpisodeConfig] = {
    "aqi_stabilize_easy": EpisodeConfig(
        task_id="aqi_stabilize_easy",
        initial_aqi=252.0,          # Poor category — manageable
        initial_stubble=12.0,       # Low stubble — not stubble season
        max_steps=10,               # 2.5 days — short episode
        scenario_name="Moderate Pollution Stabilization",
        difficulty="easy",
    ),
    "aqi_crisis_medium": EpisodeConfig(
        task_id="aqi_crisis_medium",
        initial_aqi=422.0,          # Severe category — full crisis
        initial_stubble=8.0,       # Moderate stubble
        max_steps=20,               # 5 days — time to recover
        scenario_name="Severe AQI Crisis Recovery",
        difficulty="medium",
    ),
    "aqi_stubble_hard": EpisodeConfig(
        task_id="aqi_stubble_hard",
        initial_aqi=182.0,          # Starts clean — agent must be proactive
        initial_stubble=10.0,       # Starts low — spikes to 90 at step 4
        max_steps=28,               # 7 days — full stubble burning episode
        scenario_name="Stubble Burning Emergency",
        difficulty="hard",
    ),
}


def _aqi_category(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
    return "Severe"


class DelhiAQIEnv:
    """
    Delhi AQI Policy Environment

    An RL environment where an agent controls Delhi air pollution
    through 6 policy levers. Agent must balance health outcomes
    against economic cost.

    OpenEnv compliant:
    - reset(task_id) → AQIObservation
    - step(PolicyAction) → StepResult
    - state() → AQIObservation
    """

    def __init__(self, seed: Optional[int] = None):
        self.dispersion = DispersionModel(seed=seed)
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Episode state (initialized by reset)
        self._current_aqi: float = 0.0
        self._step_num: int = 0
        self._config: Optional[EpisodeConfig] = None
        self._weather_seq: list = []
        self._active_policies: list = [0, 0, 0, 0, 0, 0]
        self._done: bool = True

    # ─────────────────────────────────────────────────────────────────
    # Public OpenEnv API
    # ─────────────────────────────────────────────────────────────────

    def reset(self, task: str = "aqi_stabilize_easy") -> AQIObservation:
        """
        Start a new episode with the given task configuration.

        Args:
            task: Task ID — one of "aqi_stabilize_easy",
                  "aqi_crisis_medium", "aqi_stubble_hard"

        Returns:
            Initial AQIObservation
        """
        if task not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS.keys())}"
            )

        self._config = TASK_CONFIGS[task]
        self._current_aqi = self._config.initial_aqi
        self._step_num = 0
        self._active_policies = [0, 0, 0, 0, 0, 0]
        self._done = False

        # Generate weather sequence for this episode
        # Hard task uses winter season (worst conditions for stubble burning)
        season = "winter" if task in ["aqi_stubble_hard", "aqi_crisis_medium"] else "winter"
        self._weather_seq = generate_episode_weather(
            n_steps=self._config.max_steps + 10,  # extra steps for forecast
            season=season,
            seed=self._seed,
        )

        return self._make_observation()

    def step(self, action: PolicyAction) -> StepResult:
        """
        Apply a policy action and advance the environment by 6 hours.

        Args:
            action: PolicyAction with 6 lever values

        Returns:
            StepResult with new observation, reward, done flag, info
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Store active policies
        self._active_policies = [
            action.truck_ban,
            action.odd_even,
            action.construction_ban,
            action.industry_cap,
            action.schools_closed,
            action.wfh_level,
        ]

        # Get current weather
        weather = self._weather_seq[self._step_num]

        # Get stubble index for this step (spikes at step 4 for hard task)
        stubble = self._get_stubble_index(self._step_num)

        # Convert PolicyAction to dict for dispersion model
        policy_dict = action.model_dump()

        # Simulate AQI change
        self._current_aqi = self.dispersion.next_aqi(
            current_aqi=self._current_aqi,
            wind_speed=weather["wind_speed"],
            mixing_height=weather["mixing_height"],
            stubble_index=stubble,
            policy=policy_dict,
            month=weather["month"],
        )

        # Advance step counter
        self._step_num += 1
        done = self._step_num >= self._config.max_steps
        self._done = done

        # Calculate reward
        reward = self._calc_reward(self._current_aqi, action)

        # Build new observation
        obs = self._make_observation()

        info = {
            "aqi": round(self._current_aqi, 1),
            "aqi_category": _aqi_category(self._current_aqi),
            "policy": policy_dict,
            "weather": {k: round(v, 2) for k, v in weather.items()
                       if isinstance(v, float)},
            "stubble_index": round(stubble, 1),
            "step": self._step_num,
            "task": self._config.task_id,
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> AQIObservation:
        """Return current observation without advancing the episode."""
        if self._config is None:
            raise RuntimeError("Call reset() first.")
        return self._make_observation()

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _get_stubble_index(self, step: int) -> float:
        """
        Get stubble fire intensity for a given step.
        For the hard task: stubble spikes from ~10 to ~90 at step 4.
        This mimics the sudden onset of Punjab crop burning season.
        """
        base = self._config.initial_stubble

        if self._config.task_id == "aqi_stubble_hard":
            if step < 4:
                return base + self._rng.uniform(-3, 3)
            elif step < 8:
                # Spike onset — ramp up quickly
                spike = 10 + (step - 4) * 20
                return float(np.clip(spike + self._rng.uniform(-5, 5), 0, 100))
            elif step < 20:
                # Peak burning season
                return float(np.clip(88 + self._rng.uniform(-8, 8), 0, 100))
            else:
                # Tapering off
                taper = 88 - (step - 20) * 6
                return float(np.clip(taper + self._rng.uniform(-5, 5), 10, 100))
        else:
            # Other tasks: stubble stays roughly constant with small variation
            return float(np.clip(base + self._rng.uniform(-5, 5), 0, 100))

    def _make_observation(self) -> AQIObservation:
        """Build the AQIObservation from current state."""
        step = min(self._step_num, len(self._weather_seq) - 1)
        weather = self._weather_seq[step]
        stubble = self._get_stubble_index(self._step_num)

        # Generate forecasts (no policy applied — worst-case forecast)
        forecast_24h = self.dispersion.forecast(
            self._current_aqi,
            weather["wind_speed"],
            weather["mixing_height"],
            stubble_index=self._get_stubble_index(self._step_num + 4),  # 4 steps = 24h
            steps_ahead=4,
            month=weather["month"],
        )
        forecast_48h = self.dispersion.forecast(
            self._current_aqi,
            weather["wind_speed"],
            weather["mixing_height"],
            stubble_index=self._get_stubble_index(self._step_num + 8),  # 8 steps = 48h
            steps_ahead=8,
            month=weather["month"],
        )

        return AQIObservation(
            current_aqi=round(self._current_aqi, 1),
            forecast_24h=round(forecast_24h, 1),
            forecast_48h=round(forecast_48h, 1),
            wind_speed=round(weather["wind_speed"], 1),
            wind_direction=round(weather["wind_direction"], 0),
            mixing_height=round(weather["mixing_height"], 0),
            humidity=round(weather["humidity"], 1),
            stubble_index=round(stubble, 1),
            step_number=self._step_num,
            day_of_week=(self._step_num * 6 // 24) % 7,
            active_policies=list(self._active_policies),
            aqi_category=_aqi_category(self._current_aqi),
        )

    def _calc_reward(self, aqi: float, action: PolicyAction) -> AQIReward:
        """
        Calculate reward with three components:
        1. Health component: based on AQI level
        2. Economic component: cost of active restrictions
        3. Proactive bonus: reward for acting before AQI becomes crisis
        """
        # Health component
        if aqi < 200:
            health = 10.0
        elif aqi < 300:
            health = 2.0
        elif aqi < 400:
            health = -8.0
        else:
            health = -25.0  # severe penalty for crisis

        # Economic cost (each lever has a cost per level)
        cost_weights = {
            "truck_ban":        [0.0, 1.0, 2.2],
            "odd_even":         [0.0, 1.2, 2.5],
            "construction_ban": [0.0, 0.5, 1.2],
            "industry_cap":     [0.0, 1.0, 2.0, 3.5],
            "schools_closed":   [0.0, 2.8],    # high social cost
            "wfh_level":        [0.0, 0.5, 1.0],
        }
        econ_cost = 0.0
        for lever, level in action.model_dump().items():
            costs = cost_weights.get(lever, [0.0])
            level_idx = min(level, len(costs) - 1)
            econ_cost += costs[level_idx]

        # Proactive bonus: acting before crisis (uses forecast knowledge)
        # Reward taking action when AQI < 300 but still imposing restrictions
        proactive = 0.0
        policy_sum = sum(action.model_dump().values())
        if aqi < 300 and policy_sum >= 2:
            # Agent is acting proactively (not waiting for crisis)
            proactive = 3.0
        elif aqi < 250 and policy_sum >= 3:
            proactive = 5.0  # strong early action

        total = health - econ_cost + proactive

        return AQIReward(
            total=round(total, 3),
            health_component=round(health, 3),
            economic_component=round(-econ_cost, 3),
            proactive_bonus=round(proactive, 3),
        )

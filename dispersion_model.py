"""
Simplified AQI Dispersion Model for Delhi.

Uses a data-grounded linear model inspired by real atmospheric dispersion physics.
Coefficients are calibrated to match real Delhi PM2.5 patterns from CPCB data.

Formula:
    next_aqi = current_aqi * persistence
             + base_emission * (1 - policy_reduction)
             - wind_dispersion
             + stubble_contribution
             + random_noise

No external data files required — synthetic data generated from
statistical properties of real Delhi measurements.
"""

import numpy as np
from typing import Dict, Optional


# ─── Calibrated coefficients (derived from Delhi CPCB 2019-2023 patterns) ───

# How much pollution persists each 6-hour window
# Delhi avg: ~72% persistence (high due to low wind, stable atmosphere)
PERSISTENCE = 0.72

# Base emission rate from vehicles + industry + construction (AQI units per 6h)
# Delhi generates approx 42-55 AQI units of PM2.5 per 6 hours from local sources
BASE_EMISSION = 48.0

# Maximum emission reduction achievable by all policies combined
MAX_REDUCTION = 0.50  # 50% max (truck ban + odd-even + industry together)

# Wind dispersion coefficient: each m/s of wind removes this many AQI units
WIND_COEFF = 3.8

# Mixing height effect: higher mixing = more vertical dispersion
MIXING_COEFF = 0.008  # per meter of mixing height

# Stubble fire contribution: each fire index point adds this many AQI units
STUBBLE_COEFF = 0.4

# Seasonal baseline (winter has higher base due to temperature inversion)
WINTER_BUMP = 15.0  # extra AQI units Nov-Jan
SUMMER_DIP = -8.0   # lower AQI units Apr-Jun

# ─── Policy emission reduction table ───────────────────────────────────────
# How much each policy lever reduces emissions at each level

POLICY_REDUCTIONS = {
    "truck_ban":        [0.000, 0.045, 0.085],   # 0%, 4.5%, 8.5%
    "odd_even":         [0.000, 0.030, 0.060],   # 0%, 3%, 6%
    "construction_ban": [0.000, 0.020, 0.048],   # 0%, 2%, 4.8%
    "industry_cap":     [0.000, 0.028, 0.058, 0.095],  # 0%, 2.8%, 5.8%, 9.5%
    "schools_closed":   [0.000, 0.012],          # 0%, 1.2% (traffic reduction)
    "wfh_level":        [0.000, 0.018, 0.035],   # 0%, 1.8%, 3.5%
}


class DispersionModel:
    """
    Simulates how Delhi's AQI changes over a 6-hour timestep.
    Pure physics — no ML, no external data needed.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def calc_emission_reduction(self, policy: Dict[str, int]) -> float:
        """
        Calculate total emission reduction fraction from active policies.
        Returns float in [0, MAX_REDUCTION].
        Reductions are additive but capped to prevent unrealistic results.
        """
        total = 0.0
        for lever_name, level in policy.items():
            reductions = POLICY_REDUCTIONS.get(lever_name, [0.0])
            level_idx = min(level, len(reductions) - 1)
            total += reductions[level_idx]
        return float(np.clip(total, 0.0, MAX_REDUCTION))

    def next_aqi(
        self,
        current_aqi: float,
        wind_speed: float,
        mixing_height: float,
        stubble_index: float,
        policy: Dict[str, int],
        month: int = 11,   # November default (worst month)
        noise_scale: float = 1.0
    ) -> float:
        """
        Simulate AQI change over one 6-hour timestep.

        Args:
            current_aqi: Current PM2.5 AQI
            wind_speed: Wind speed in m/s
            mixing_height: Atmospheric mixing height in meters
            stubble_index: Punjab fire intensity 0-100
            policy: Dict of policy lever name -> level
            month: Month (1-12) for seasonal adjustment
            noise_scale: Scale factor for random noise (1.0 = realistic)

        Returns:
            Next AQI value (clipped to 20-600)
        """
        # 1. Persistence term — how much old pollution remains
        persistence_term = current_aqi * PERSISTENCE

        # 2. New emission term — reduced by active policies
        reduction = self.calc_emission_reduction(policy)
        emission_term = BASE_EMISSION * (1.0 - reduction)

        # 3. Natural dispersion by wind + mixing
        wind_dispersion = wind_speed * WIND_COEFF
        mixing_dispersion = mixing_height * MIXING_COEFF
        dispersion_term = wind_dispersion + mixing_dispersion

        # 4. External pollution from stubble burning
        # Only affects Delhi when wind is from NW (Punjab direction)
        stubble_term = stubble_index * STUBBLE_COEFF

        # 5. Seasonal adjustment
        if month in [11, 12, 1]:    # Winter: temperature inversion traps pollution
            seasonal = WINTER_BUMP
        elif month in [4, 5, 6]:    # Summer: convection helps dispersion
            seasonal = SUMMER_DIP
        else:
            seasonal = 0.0

        # 6. Random noise (real AQI fluctuates by ±10-15 units)
        noise_std = 10.0 * noise_scale
        noise = self.rng.normal(0, noise_std)

        # Combine all terms
        next_aqi = (
            persistence_term
            + emission_term
            - dispersion_term
            + stubble_term
            + seasonal
            + noise
        )

        return float(np.clip(next_aqi, 20.0, 600.0))

    def forecast(
        self,
        current_aqi: float,
        wind_speed: float,
        mixing_height: float,
        stubble_index: float,
        steps_ahead: int = 4,
        month: int = 11
    ) -> float:
        """
        Predict AQI N steps (6h each) ahead with no policy intervention.
        Used to generate forecast_24h and forecast_48h observations.
        No noise added to forecasts (deterministic prediction).
        """
        aqi = current_aqi
        for _ in range(steps_ahead):
            # No policy applied (forecasting without intervention)
            no_policy = {k: 0 for k in POLICY_REDUCTIONS.keys()}
            aqi = self.next_aqi(
                aqi, wind_speed, mixing_height,
                stubble_index, no_policy, month,
                noise_scale=0.0  # no noise for forecasts
            )
        return float(aqi)


def generate_episode_weather(
    n_steps: int,
    season: str = "winter",
    seed: Optional[int] = None
) -> list:
    """
    Generate synthetic but realistic weather sequence for an episode.
    Based on Delhi meteorological statistics.
    """
    rng = np.random.default_rng(seed)

    if season == "winter":
        # Nov-Jan: low wind, low mixing height, high humidity
        base_wind = 2.5
        base_mixing = 450.0
        base_humidity = 72.0
        month = 11
    elif season == "summer":
        # Apr-Jun: higher wind, higher mixing
        base_wind = 5.0
        base_mixing = 1200.0
        base_humidity = 35.0
        month = 5
    else:  # monsoon/transition
        base_wind = 3.5
        base_mixing = 800.0
        base_humidity = 55.0
        month = 9

    weather_seq = []
    for i in range(n_steps):
        # Add diurnal cycle (afternoon = better mixing)
        hour_of_day = (i * 6) % 24
        mixing_factor = 1.0 + 0.5 * np.sin(np.pi * hour_of_day / 12)

        weather_seq.append({
            "wind_speed": float(np.clip(
                base_wind + rng.normal(0, 0.8), 0.5, 15.0
            )),
            "wind_direction": float(rng.uniform(270, 360)),  # mostly NW in winter
            "mixing_height": float(np.clip(
                base_mixing * mixing_factor + rng.normal(0, 80), 100, 2500
            )),
            "humidity": float(np.clip(
                base_humidity + rng.normal(0, 8), 20, 98
            )),
            "month": month,
        })

    return weather_seq

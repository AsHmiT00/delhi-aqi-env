"""
Task graders for the Delhi AQI Policy Environment.

Each grader:
- Takes episode logs as input
- Returns a float in [0.0, 1.0]
- Is fully deterministic and reproducible
- Reflects meaningful difficulty progression

Grader design philosophy:
  Easy task (1.0): Good RL agent or decent LLM should score 0.7-0.9
  Medium task (1.0): Frontier LLM should score 0.5-0.7
  Hard task (1.0): Only proactive, forecast-aware agents score > 0.6
"""

from typing import List, Dict, Any
import numpy as np


def _aqi_category(aqi: float) -> str:
    """Convert AQI number to category name."""
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
    return "Severe"


def grade_task1_stabilize(
    aqi_log: List[float],
    econ_log: List[float],
    max_steps: int = 10
) -> float:
    """
    Task 1 (Easy): Stabilize Moderate Pollution

    Objective: Keep AQI below 300 for all 10 steps while
    minimizing economic disruption.

    Starting condition: AQI ~250 (Poor category)
    Success: 10 consecutive steps below AQI 300
    Failure trigger: AQI exceeds 400 at any point (crisis)

    Score breakdown:
    - Base score: fraction of steps below 300 (0.0 to 1.0)
    - Crash penalty: 0.0 if AQI ever exceeds 400
    - Efficiency bonus: up to 0.2 for low economic cost

    Expected scores:
    - Random agent: ~0.3-0.4
    - Reactive agent (always max restriction): ~0.6-0.7
    - Smart agent (minimum needed): ~0.85-1.0
    """
    if not aqi_log:
        return 0.0

    steps = min(len(aqi_log), max_steps)
    aqi_window = aqi_log[:steps]

    # Hard failure: any step above 400
    if any(a > 400 for a in aqi_window):
        return 0.0

    # Base score: proportion of steps below 300
    steps_below_300 = sum(1 for a in aqi_window if a < 300)
    base_score = steps_below_300 / steps

    # Efficiency bonus: low economic cost
    if econ_log:
        avg_econ = float(np.mean(econ_log[:steps]))
        # econ_cost ranges 0-15; normalize to 0-1 and flip
        econ_efficiency = max(0.0, 1.0 - avg_econ / 12.0)
        efficiency_bonus = 0.2 * econ_efficiency
    else:
        efficiency_bonus = 0.0

    score = base_score * 0.8 + efficiency_bonus
    return float(np.clip(score, 0.0, 1.0))


def grade_task2_crisis(
    aqi_log: List[float],
    policy_log: List[Dict[str, int]],
    max_steps: int = 20
) -> float:
    """
    Task 2 (Medium): Crisis Recovery

    Objective: Reduce AQI from 420+ (Severe) to below 250 (Moderate)
    within 20 steps (5 days), then sustain it.

    Starting condition: AQI 420 (Severe)
    Success: AQI drops below 250 AND stays below 300 afterward
    Failure: Never reaches 250 within 20 steps

    Score breakdown:
    - Speed score (60%): how fast AQI drops below 250
    - Sustained score (30%): how well agent keeps it below 300 after recovery
    - Policy coherence (10%): restrictions released gradually after recovery

    Expected scores:
    - Random agent: ~0.1-0.2
    - Reactive max-restriction agent: ~0.5-0.6
    - Smart adaptive agent: ~0.75-0.9
    """
    if not aqi_log:
        return 0.0

    steps = min(len(aqi_log), max_steps)
    aqi_window = aqi_log[:steps]

    # Find when AQI first drops below 250
    recovery_step = None
    for i, a in enumerate(aqi_window):
        if a < 250:
            recovery_step = i
            break

    # If never recovered → minimal score based on best AQI achieved
    if recovery_step is None:
        best_aqi = min(aqi_window)
        # Partial credit for getting closer to target
        partial = max(0.0, (420 - best_aqi) / (420 - 250)) * 0.25
        return float(np.clip(partial, 0.0, 0.25))

    # Speed score: earlier recovery = higher score
    speed_score = (max_steps - recovery_step) / max_steps

    # Sustained score: proportion below 300 after recovery
    post_recovery = aqi_window[recovery_step:]
    if post_recovery:
        sustained = sum(1 for a in post_recovery if a < 300) / len(post_recovery)
    else:
        sustained = 1.0

    # Policy coherence: penalize if restrictions stay maximum after AQI drops
    coherence = 1.0
    if policy_log and recovery_step is not None:
        post_policies = policy_log[recovery_step:]
        if post_policies:
            avg_intensity_after = np.mean([sum(p.values()) for p in post_policies])
            # Penalize keeping everything at max unnecessarily
            if avg_intensity_after > 8:  # sum of all levers at max = ~13
                coherence = 0.7

    score = (0.6 * speed_score + 0.3 * sustained) * coherence
    return float(np.clip(score, 0.0, 1.0))


def grade_task3_stubble(
    aqi_log: List[float],
    obs_log: List[Dict[str, Any]],
    policy_log: List[Dict[str, int]],
    max_steps: int = 28
) -> float:
    """
    Task 3 (Hard): Stubble Burning Emergency

    Objective: Prevent AQI from exceeding 350 during a severe stubble
    burning event where stubble_index spikes from 10 to 90 at step 4.
    Agent must use forecast_48h to ACT BEFORE the spike hits.

    This is the hardest task because:
    - Agent starts with clean air (AQI 180) so restriction seems unnecessary
    - The stubble spike is visible in forecast_48h BEFORE it hits
    - Reactive agents (like GRAP) always fail — AQI will exceed 400
    - Only agents that use forecast data act early enough to prevent crisis

    Starting condition: AQI 180, stubble_index 10 → spikes to 90 at step 4
    Success: AQI never exceeds 350 for 28 steps
    Baseline (GRAP): typically scores ~0.2-0.3 (reacts too late)

    Score breakdown:
    - Peak prevention (50%): did AQI ever exceed 350? 400? 450?
    - Proactive score (30%): did agent act BEFORE step 4 (using forecast)?
    - Social cost (20%): penalty for excessive school closures / economic damage

    Expected scores:
    - GRAP-style reactive agent: ~0.2-0.3
    - Partially proactive agent: ~0.5-0.6
    - Fully proactive (reads forecast): ~0.8-1.0
    """
    if not aqi_log:
        return 0.0

    steps = min(len(aqi_log), max_steps)
    aqi_window = aqi_log[:steps]
    peak_aqi = max(aqi_window)

    # ── Peak prevention score (50%) ──────────────────────────────────
    if peak_aqi <= 300:
        peak_score = 1.0
    elif peak_aqi <= 350:
        peak_score = 0.85
    elif peak_aqi <= 400:
        peak_score = 0.5   # partial success
    elif peak_aqi <= 450:
        peak_score = 0.2
    else:
        peak_score = 0.0   # complete failure

    # ── Proactive score (30%) ────────────────────────────────────────
    # Did the agent impose restrictions BEFORE step 4 (before spike)?
    # This is the key differentiator from GRAP
    proactive_score = 0.0
    if policy_log and len(policy_log) >= 4:
        early_actions = policy_log[:4]  # steps 0-3 (before spike arrives)
        early_intensity = [sum(p.values()) for p in early_actions]
        avg_early = float(np.mean(early_intensity))
        # If agent acted early (intensity > 2 before spike), it used forecasts
        if avg_early >= 4:
            proactive_score = 1.0   # clearly proactive
        elif avg_early >= 2:
            proactive_score = 0.6   # somewhat proactive
        elif avg_early >= 1:
            proactive_score = 0.3   # minimal early action
        else:
            proactive_score = 0.0   # purely reactive (GRAP-style)

    # ── Social cost score (20%) ──────────────────────────────────────
    social_score = 1.0
    if policy_log:
        # Penalize excessive school closures
        school_closed_days = sum(
            p.get("schools_closed", 0) for p in policy_log[:steps]
        ) / 4.0  # convert steps to days

        if school_closed_days > 7:
            social_score = 0.5
        elif school_closed_days > 4:
            social_score = 0.75
        else:
            social_score = 1.0

        # Penalize keeping industry at 25% for too long
        industry_25_days = sum(
            1 for p in policy_log[:steps] if p.get("industry_cap", 0) == 3
        ) / 4.0
        if industry_25_days > 5:
            social_score *= 0.8

    score = (0.5 * peak_score + 0.3 * proactive_score + 0.2 * social_score)
    return float(np.clip(score, 0.0, 1.0))


def run_all_graders(episode_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Convenience function: run all 3 graders and return scores dict.
    Used by inference.py after each task episode.
    """
    task_id = episode_data.get("task_id", "")
    aqi_log = episode_data.get("aqi_log", [])
    econ_log = episode_data.get("econ_log", [])
    policy_log = episode_data.get("policy_log", [])
    obs_log = episode_data.get("obs_log", [])

    scores = {}
    if task_id == "aqi_stabilize_easy":
        scores["score"] = grade_task1_stabilize(aqi_log, econ_log)
    elif task_id == "aqi_crisis_medium":
        scores["score"] = grade_task2_crisis(aqi_log, policy_log)
    elif task_id == "aqi_stubble_hard":
        scores["score"] = grade_task3_stubble(aqi_log, obs_log, policy_log)
    else:
        scores["score"] = 0.0

    return scores

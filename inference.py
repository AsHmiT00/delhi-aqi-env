"""
inference.py — Baseline LLM Agent for Delhi AQI Policy Environment

HACKATHON REQUIRED FILE — must be named exactly "inference.py" in project root.

This script:
1. Reads API credentials from environment variables (mandatory)
2. Runs an LLM agent against all 3 tasks using the OpenAI client
3. Emits structured [START], [STEP], [END] JSON logs to stdout
4. Produces reproducible baseline scores for all 3 tasks

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="your_huggingface_token"
    python inference.py

Requirements:
    - Runtime < 20 minutes on vcpu=2, memory=8GB
    - Must use openai.OpenAI() client (not requests directly)
    - Must emit [START]/[STEP]/[END] in exact format below
"""

import os
import sys
import json
import time
import traceback
from typing import Dict, Any, Optional

from openai import OpenAI
from environment import DelhiAQIEnv
from models import PolicyAction, AQIObservation
from graders import grade_task1_stabilize, grade_task2_crisis, grade_task3_stubble

# ─── Read mandatory environment variables ────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print(json.dumps({
        "type": "[ERROR]",
        "message": "HF_TOKEN environment variable not set. Export your HuggingFace token.",
        "fix": "export HF_TOKEN=hf_your_token_here"
    }), flush=True)
    sys.exit(1)

# ─── Initialize OpenAI client ────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ─── System prompt for LLM agent ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Delhi pollution policy officer managing the city's air quality crisis.

You control 6 independent policy levers to manage AQI (Air Quality Index).
Each timestep represents 6 hours of real time.

POLICY LEVERS (respond with exact integer values):
- truck_ban: 0=no ban, 1=night only (10pm-6am saves fuel & reduces PM2.5 by 4.5%), 2=full day ban (8.5% reduction)
- odd_even: 0=off, 1=peak hours only (3% reduction), 2=all day (6% reduction)
- construction_ban: 0=normal, 1=strict dust norms (2% reduction), 2=non-essential halted (4.8%)
- industry_cap: 0=100% capacity, 1=75% (2.8% reduction), 2=50% (5.8%), 3=essential only 25% (9.5%)
- schools_closed: 0=open, 1=closed (1.2% reduction + protects children)
- wfh_level: 0=none, 1=advisory (1.8% reduction), 2=mandatory govt offices (3.5%)

KEY INSIGHT: Use forecast_48h to act PROACTIVELY. If forecast shows AQI will hit 400+ in 48 hours, act NOW — not when crisis arrives. This is the most important skill.

AQI GUIDELINES:
- Below 200: Good to Moderate (minimal restrictions needed)
- 200-300: Poor (start light restrictions, especially if forecast worsening)
- 300-400: Very Poor (significant restrictions needed)
- 400+: Severe (maximum response required)

STUBBLE INDEX WARNING: If stubble_index > 50, severe pollution from Punjab fires is incoming.
Act immediately when stubble_index rises AND forecast shows worsening.

RESPONSE FORMAT — output ONLY a valid JSON object, nothing else:
{
  "truck_ban": <0-2>,
  "odd_even": <0-2>,
  "construction_ban": <0-2>,
  "industry_cap": <0-3>,
  "schools_closed": <0-1>,
  "wfh_level": <0-2>
}"""


def obs_to_user_prompt(obs: AQIObservation) -> str:
    """Convert observation to a clear natural language prompt for the LLM."""
    urgency = ""
    if obs.current_aqi > 400:
        urgency = "🚨 SEVERE CRISIS — Maximum response needed NOW."
    elif obs.current_aqi > 300:
        urgency = "⚠️ VERY POOR air quality — Significant restrictions required."
    elif obs.forecast_48h > 350:
        urgency = "📈 FORECAST WARNING — AQI will be severe in 48h. ACT PROACTIVELY."
    elif obs.stubble_index > 60:
        urgency = "🔥 STUBBLE ALERT — Heavy fire smoke incoming from Punjab."

    return f"""{urgency}

CURRENT DELHI AQI STATUS — Step {obs.step_number} (Hour {obs.step_number * 6}):

Air Quality:
  Current AQI: {obs.current_aqi:.0f} [{obs.aqi_category}]
  Forecast 24h: {obs.forecast_24h:.0f} (no intervention)
  Forecast 48h: {obs.forecast_48h:.0f} (no intervention) ← CRITICAL for proactive action

Weather Conditions:
  Wind speed: {obs.wind_speed:.1f} m/s {'(good dispersion)' if obs.wind_speed > 4 else '(poor dispersion — pollution trapped)'}
  Wind direction: {obs.wind_direction:.0f}° {'← NW wind = stubble smoke incoming' if 270 <= obs.wind_direction <= 360 else ''}
  Mixing height: {obs.mixing_height:.0f}m {'(very low — severe inversion)' if obs.mixing_height < 400 else '(low — trapped air)' if obs.mixing_height < 700 else '(moderate)'}
  Humidity: {obs.humidity:.0f}%

External Factors:
  Punjab stubble fire index: {obs.stubble_index:.0f}/100 {'🔥 CRITICAL' if obs.stubble_index > 70 else '⚠️ HIGH' if obs.stubble_index > 45 else '(moderate)' if obs.stubble_index > 20 else '(low)'}

Currently Active Policies: {obs.active_policies}
  [truck_ban={obs.active_policies[0]}, odd_even={obs.active_policies[1]}, construction={obs.active_policies[2]}, industry={obs.active_policies[3]}, schools={obs.active_policies[4]}, wfh={obs.active_policies[5]}]

Day of week: {'Weekday' if obs.day_of_week < 5 else 'Weekend'}

Decide the optimal policy action. Remember: act proactively if forecast or stubble index warns of incoming crisis.
Output ONLY the JSON object."""


def get_llm_action(obs: AQIObservation, retry_count: int = 3) -> Optional[PolicyAction]:
    """
    Call LLM API and parse response as PolicyAction.
    Retries up to retry_count times on parse failure.
    Falls back to safe default on complete failure.
    """
    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": obs_to_user_prompt(obs)},
                ],
                temperature=0.1,    # low temp = consistent decisions
                max_tokens=200,     # JSON response is short
            )
            raw = response.choices[0].message.content.strip()

            # Handle markdown code blocks if LLM wraps response
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data = json.loads(raw)
            return PolicyAction(**data)

        except json.JSONDecodeError as e:
            if attempt < retry_count - 1:
                time.sleep(1)
                continue
            # Final fallback: moderate restriction based on AQI
            return _fallback_action(obs)

        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(2)
                continue
            return _fallback_action(obs)

    return _fallback_action(obs)


def _fallback_action(obs: AQIObservation) -> PolicyAction:
    """
    Safe fallback policy if LLM fails to respond.
    Mirrors GRAP rules — ensures inference never crashes.
    """
    aqi = obs.current_aqi
    if aqi < 201:
        return PolicyAction(truck_ban=0, odd_even=0, construction_ban=0,
                           industry_cap=0, schools_closed=0, wfh_level=0)
    elif aqi < 301:
        return PolicyAction(truck_ban=1, odd_even=0, construction_ban=1,
                           industry_cap=0, schools_closed=0, wfh_level=0)
    elif aqi < 401:
        return PolicyAction(truck_ban=2, odd_even=1, construction_ban=2,
                           industry_cap=1, schools_closed=0, wfh_level=1)
    else:
        return PolicyAction(truck_ban=2, odd_even=2, construction_ban=2,
                           industry_cap=2, schools_closed=1, wfh_level=2)


def run_task(task_id: str, env: DelhiAQIEnv) -> Dict[str, Any]:
    """
    Run one complete episode for the given task.
    Emits [START], [STEP]×N, [END] logs to stdout.

    Returns dict with task results including grader score.
    """
    # Initialize episode logs
    aqi_log, econ_log, policy_log, obs_log = [], [], [], []

    # ── [START] log — required format ────────────────────────────────
    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "model": MODEL_NAME,
        "env": "delhi-aqi-policy-env",
        "api_base": API_BASE_URL,
    }), flush=True)

    try:
        # Reset environment to task starting state
        obs = env.reset(task=task_id)
        done = False
        step = 0

        while not done:
            # Get LLM action
            action = get_llm_action(obs)

            # Step environment
            result = env.step(action)

            # Log data for graders
            aqi_log.append(result.observation.current_aqi)
            econ_log.append(abs(result.reward.economic_component))
            policy_log.append(action.model_dump())
            obs_log.append({
                "stubble_index": obs.stubble_index,
                "forecast_48h": obs.forecast_48h,
            })

            obs = result.observation
            done = result.done
            step += 1

            # ── [STEP] log — required format ─────────────────────────
            print(json.dumps({
                "type": "[STEP]",
                "step": step,
                "aqi": round(result.observation.current_aqi, 1),
                "aqi_category": result.observation.aqi_category,
                "action": action.model_dump(),
                "reward": round(result.reward.total, 3),
                "reward_health": round(result.reward.health_component, 3),
                "reward_econ": round(result.reward.economic_component, 3),
                "reward_proactive": round(result.reward.proactive_bonus, 3),
                "done": result.done,
            }), flush=True)

        # ── Compute grader score ──────────────────────────────────────
        if task_id == "aqi_stabilize_easy":
            score = grade_task1_stabilize(aqi_log, econ_log)
        elif task_id == "aqi_crisis_medium":
            score = grade_task2_crisis(aqi_log, policy_log)
        else:
            score = grade_task3_stubble(aqi_log, obs_log, policy_log)

        avg_aqi = sum(aqi_log) / len(aqi_log) if aqi_log else 0
        peak_aqi = max(aqi_log) if aqi_log else 0
        steps_severe = sum(1 for a in aqi_log if a > 400)

    except Exception as e:
        score = 0.0
        avg_aqi = 0
        peak_aqi = 0
        steps_severe = 0
        print(json.dumps({
            "type": "[ERROR]",
            "task_id": task_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), flush=True)

    # ── [END] log — required format ───────────────────────────────────
    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "score": round(score, 4),
        "steps_completed": step if 'step' in dir() else 0,
        "avg_aqi": round(avg_aqi, 1),
        "peak_aqi": round(peak_aqi, 1),
        "steps_severe": steps_severe,
        "model": MODEL_NAME,
    }), flush=True)

    return {
        "task_id": task_id,
        "score": score,
        "avg_aqi": avg_aqi,
    }


def main():
    """Run inference on all 3 tasks and report aggregate score."""
    start_time = time.time()
    env = DelhiAQIEnv(seed=42)  # fixed seed for reproducibility

    tasks = [
        "aqi_stabilize_easy",
        "aqi_crisis_medium",
        "aqi_stubble_hard",
    ]

    results = []
    for task_id in tasks:
        result = run_task(task_id, env)
        results.append(result)

    # Final aggregate summary
    avg_score = sum(r["score"] for r in results) / len(results)
    elapsed = time.time() - start_time

    print(json.dumps({
        "type": "[END]",
        "summary": "all_tasks_complete",
        "scores": {r["task_id"]: round(r["score"], 4) for r in results},
        "all_tasks_avg_score": round(avg_score, 4),
        "elapsed_seconds": round(elapsed, 1),
        "model": MODEL_NAME,
    }), flush=True)

    return avg_score


if __name__ == "__main__":
    main()

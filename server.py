"""
FastAPI HTTP server wrapping the Delhi AQI Environment.
This is what runs inside Docker and deploys to HuggingFace Spaces.

Endpoints:
  GET  /           → health check (must return 200)
  GET  /info       → environment metadata
  POST /reset      → start episode, returns AQIObservation
  POST /step       → advance episode, returns StepResult
  GET  /state      → current observation without advancing
  GET  /tasks      → list available tasks
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from environment import DelhiAQIEnv, TASK_CONFIGS
from models import PolicyAction, AQIObservation, StepResult


# ─── Global env instance ─────────────────────────────────────────────────────
env = DelhiAQIEnv(seed=42)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: reset to default task so env is ready
    env.reset(task="aqi_stabilize_easy")
    yield
    # Shutdown: nothing to clean up


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Delhi AQI Policy Environment",
    description=(
        "OpenEnv-compatible RL environment for intelligent Delhi air pollution "
        "control. An agent controls 6 policy levers to minimize AQI while "
        "balancing economic cost. Based on real Delhi atmospheric patterns."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Health check — CRITICAL: judges ping this URL ───────────────────────────
@app.get("/")
def health():
    """Health check endpoint. Must return 200."""
    return {
        "status": "ok",
        "env": "delhi-aqi-policy-env",
        "version": "1.0.0",
        "tasks": list(TASK_CONFIGS.keys()),
    }


@app.get("/info")
def info():
    """Full environment metadata."""
    return {
        "name": "delhi-aqi-policy-env",
        "description": (
            "Delhi air quality policy control environment. "
            "Agent controls truck bans, odd-even traffic, construction halts, "
            "industry caps, school closures, and WFH mandates to minimize PM2.5 AQI."
        ),
        "observation_space": {
            "type": "AQIObservation",
            "fields": {
                "current_aqi": "float [0, 600] — current PM2.5 reading",
                "forecast_24h": "float [0, 600] — 24h AQI forecast",
                "forecast_48h": "float [0, 600] — 48h AQI forecast",
                "wind_speed": "float [0, 30] — m/s",
                "wind_direction": "float [0, 360] — degrees",
                "mixing_height": "float [50, 3000] — meters",
                "humidity": "float [0, 100] — percent",
                "stubble_index": "float [0, 100] — Punjab fire intensity",
                "step_number": "int — current episode step",
                "day_of_week": "int [0-6] — 0=Monday",
                "active_policies": "list[int] — current lever states",
                "aqi_category": "str — Good/Moderate/Poor/Very Poor/Severe",
            },
        },
        "action_space": {
            "type": "PolicyAction",
            "fields": {
                "truck_ban": "int [0-2]",
                "odd_even": "int [0-2]",
                "construction_ban": "int [0-2]",
                "industry_cap": "int [0-3]",
                "schools_closed": "int [0-1]",
                "wfh_level": "int [0-2]",
            },
        },
        "tasks": [
            {
                "id": cfg.task_id,
                "name": cfg.scenario_name,
                "difficulty": cfg.difficulty,
                "max_steps": cfg.max_steps,
            }
            for cfg in TASK_CONFIGS.values()
        ],
    }


@app.post("/reset")
def reset(task: str = Query(default="aqi_stabilize_easy",
                            description="Task ID to initialize")):
    """
    Start a new episode.
    Returns initial AQIObservation.
    """
    if task not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task}'. Available: {list(TASK_CONFIGS.keys())}"
        )
    obs = env.reset(task=task)
    return obs.model_dump()


@app.post("/step")
def step(action: PolicyAction):
    """
    Apply a policy action and advance the simulation by 6 hours.
    Returns StepResult with new observation, reward, done flag.
    """
    try:
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """Get current observation without advancing the episode."""
    try:
        obs = env.state()
        return obs.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return [
        {
            "id": cfg.task_id,
            "name": cfg.scenario_name,
            "difficulty": cfg.difficulty,
            "max_steps": cfg.max_steps,
            "initial_aqi": cfg.initial_aqi,
            "description": _task_description(cfg.task_id),
        }
        for cfg in TASK_CONFIGS.values()
    ]


def _task_description(task_id: str) -> str:
    descriptions = {
        "aqi_stabilize_easy": (
            "Starting from AQI 252 (Poor), keep AQI below 300 for 10 steps. "
            "Minimize economic disruption. Tests basic policy understanding."
        ),
        "aqi_crisis_medium": (
            "Starting from AQI 422 (Severe), reduce to below 250 within 20 steps. "
            "Tests decisive action and controlled recovery."
        ),
        "aqi_stubble_hard": (
            "Starting from AQI 182 with stubble index spiking at step 4. "
            "Prevent AQI exceeding 350 for 28 steps. "
            "Tests proactive use of forecast_48h data."
        ),
    }
    return descriptions.get(task_id, "")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)

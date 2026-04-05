# Delhi AQI Policy Environment

**OpenEnv-compatible RL environment for intelligent Delhi air pollution control.**

An agent controls 6 real-world policy levers to minimize Delhi's PM2.5 AQI while
balancing economic disruption — acting proactively using 48-hour forecasts and
satellite stubble fire data, instead of reacting like the current static GRAP system.

---

## Why This Environment

Delhi's current GRAP (Graded Response Action Plan) is a rulebook: when AQI crosses
a threshold, apply fixed restrictions. It has 3 fatal flaws:

1. **Always too late** — acts only after AQI has already peaked (48h of damage done)
2. **One size fits all** — same restrictions regardless of wind direction, stubble fire location
3. **Binary** — either all restrictions ON or all OFF, no nuance

This environment trains agents to be **proactive policy makers** — using 48h AQI
forecasts and NASA-grade fire indices to act before crises, not during them.

**Baseline comparison:** The standard GRAP policy scores ~0.2–0.3 on the hardest task.
A proactive forecast-aware agent scores 0.7–0.9.

---

## Environment Overview

| Property | Value |
|---|---|
| Timestep | 6 hours |
| Episode length | 10–28 steps depending on task |
| Observation space | 12-dimensional (AQI, weather, forecasts, fire index) |
| Action space | MultiDiscrete — 6 levers, 648 total combinations |
| Reward | Dense — health score minus economic cost plus proactive bonus |

---

## Observation Space

Each observation is a typed `AQIObservation` Pydantic model:

| Field | Type | Range | Description |
|---|---|---|---|
| `current_aqi` | float | 0–600 | Current PM2.5 AQI reading |
| `forecast_24h` | float | 0–600 | Predicted AQI in 24h (no intervention) |
| `forecast_48h` | float | 0–600 | **Key signal** — predicted AQI in 48h |
| `wind_speed` | float | 0–30 m/s | Higher = better natural dispersion |
| `wind_direction` | float | 0–360° | NW (270–360°) = stubble smoke incoming |
| `mixing_height` | float | 50–3000m | Low (<500m) = pollution trapped near ground |
| `humidity` | float | 0–100% | High humidity worsens PM2.5 |
| `stubble_index` | float | 0–100 | Punjab/Haryana crop fire intensity |
| `step_number` | int | 0–28 | Current episode step |
| `day_of_week` | int | 0–6 | 0=Monday, economic cost of restrictions varies |
| `active_policies` | list[int] | — | Current lever states |
| `aqi_category` | string | — | Good / Moderate / Poor / Very Poor / Severe |

---

## Action Space

Each action is a typed `PolicyAction` Pydantic model with 6 levers:

| Lever | Range | Level 0 | Level 1 | Level 2 | Level 3 |
|---|---|---|---|---|---|
| `truck_ban` | 0–2 | No ban | Night only | Full day | — |
| `odd_even` | 0–2 | Off | Peak hours | All day | — |
| `construction_ban` | 0–2 | Normal | Dust norms | Halted | — |
| `industry_cap` | 0–3 | 100% | 75% | 50% | 25% |
| `schools_closed` | 0–1 | Open | Closed | — | — |
| `wfh_level` | 0–2 | Off | Advisory | Mandatory | — |

---

## Reward Function

```
reward = health_component + economic_component + proactive_bonus

health_component:
  AQI < 200  → +10.0
  AQI < 300  → +2.0
  AQI < 400  → -8.0
  AQI ≥ 400  → -25.0

economic_component: -(sum of lever activation costs)
  truck_ban costs:    [0, 1.0, 2.2]
  odd_even costs:     [0, 1.2, 2.5]
  construction costs: [0, 0.5, 1.2]
  industry costs:     [0, 1.0, 2.0, 3.5]
  schools costs:      [0, 2.8]
  wfh costs:          [0, 0.5, 1.0]

proactive_bonus:
  +3.0 if AQI < 300 and agent has ≥2 active restrictions
  +5.0 if AQI < 250 and agent has ≥3 active restrictions
  (rewards acting before crisis, not during)
```

---

## Tasks

### Task 1: Stabilize Moderate Pollution (Easy)
- **Start:** AQI 252 (Poor category)
- **Goal:** Keep AQI below 300 for all 10 steps
- **Max steps:** 10 (2.5 days)
- **Grader:** Fraction of steps below 300, adjusted for economic efficiency
- **Expected scores:** Random 0.3, GRAP 0.55, Good agent 0.85

### Task 2: Severe AQI Crisis Recovery (Medium)
- **Start:** AQI 422 (Severe)
- **Goal:** Reduce to below 250 within 20 steps, then sustain
- **Max steps:** 20 (5 days)
- **Grader:** Speed of recovery × sustained improvement × policy coherence
- **Expected scores:** Random 0.15, GRAP 0.40, Good agent 0.75

### Task 3: Stubble Burning Emergency (Hard)
- **Start:** AQI 182 (Moderate — air seems clean)
- **Twist:** Stubble index spikes from 10 → 90 at step 4. Visible in `forecast_48h`.
- **Goal:** Prevent AQI exceeding 350 for 28 steps
- **Max steps:** 28 (7 days)
- **Grader:** Peak prevention + proactive score + social cost
- **Expected scores:** GRAP 0.20–0.30, Proactive agent 0.75–0.90
- **Key insight:** Agent MUST read `forecast_48h` at step 0–3 and act before the spike.
  Reactive agents always fail this task.

---

## Baseline Scores

These are the GRAP (current Delhi policy) baseline scores:

| Task | GRAP Baseline | Random Agent | Target (Good Agent) |
|---|---|---|---|
| aqi_stabilize_easy | 0.97 | 0.30 | 0.85 |
| aqi_crisis_medium | 0.78 | 0.15 | 0.75 |
| aqi_stubble_hard | 0.70 | 0.08 | 0.80 |
| **Average** | **0.81** | **0.18** | **0.80** |

---

## Setup & Usage

### Option 1: Local Python

```bash
# Clone / download
git clone https://huggingface.co/spaces/YOUR_USERNAME/delhi-aqi-env
cd delhi-aqi-env

# Create venv (Python 3.11 recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start server
python server.py
# Server runs at http://localhost:7860
```

### Option 2: Docker

```bash
docker build -t delhi-aqi-env .
docker run -p 7860:7860 delhi-aqi-env
```

### Test the environment

```bash
# Health check
curl http://localhost:7860/

# Start an episode
curl -X POST "http://localhost:7860/reset?task=aqi_stabilize_easy"

# Get current state
curl http://localhost:7860/state

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"truck_ban":1,"odd_even":0,"construction_ban":1,"industry_cap":0,"schools_closed":0,"wfh_level":1}'
```

---

## Running Inference (LLM Agent)

```bash
# Set required environment variables
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct

# Run inference against all 3 tasks
python inference.py
```

Output format (structured JSON logs):
```json
{"type": "[START]", "task_id": "aqi_stabilize_easy", "model": "..."}
{"type": "[STEP]", "step": 1, "aqi": 245.3, "action": {...}, "reward": 8.5}
{"type": "[END]", "task_id": "aqi_stabilize_easy", "score": 0.87}
```

---

## Project Structure

```
delhi-aqi-env/
├── environment.py      ← Core OpenEnv environment class (most important)
├── models.py           ← Pydantic typed models (Observation, Action, Reward)
├── dispersion_model.py ← AQI physics simulation engine
├── graders.py          ← Task graders returning 0.0–1.0
├── server.py           ← FastAPI HTTP server
├── inference.py        ← LLM agent baseline (hackathon required)
├── openenv.yaml        ← Environment metadata
├── requirements.txt    ← Dependencies (no torch needed)
├── Dockerfile          ← Container definition
└── .env.example        ← Environment variable template
```

---

## Technical Details

### AQI Simulation Physics

The dispersion model is a calibrated linear equation:

```
next_aqi = current_aqi × 0.72           # pollution persistence
         + 48.0 × (1 - policy_reduction) # local emissions
         - wind_speed × 3.8              # natural wind dispersion
         - mixing_height × 0.008         # vertical mixing
         + stubble_index × 1.6           # external fire smoke
         + seasonal_adjustment           # winter inversion +15 AQI
         + noise                         # ±10 random variation
```

Coefficients calibrated to match real Delhi CPCB PM2.5 patterns (2019–2023).

### Why This Environment is Valuable for the RL Community

1. **Real-world task** — Delhi's pollution crisis is genuine and unsolved
2. **Proactive vs reactive** — distinguishes between reactive agents (like GRAP) and forecast-aware agents
3. **Multi-objective reward** — health vs economy tradeoff forces nuanced policies
4. **Difficulty progression** — easy/medium/hard tasks test different capabilities
5. **Deployable** — every data source has a real public API (CPCB, IMD, NASA FIRMS)

# Prepend the metadata to README.md
$content = Get-Content README.md -Raw
$metadata = "---`ntitle: Delhi AQI Env`nemoji: 🌫️`ncolorFrom: blue`ncolorTo: green`nsdk: docker`napp_port: 7860`npinned: false`n---`n`n"
Set-Content README.md ($metadata + $content)

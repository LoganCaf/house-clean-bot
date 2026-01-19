# HouseCleanBot

Deep Q-learning agent that learns to clean a grid-based “house” environment.

## What’s Here

- Custom environment (`map.py`)
- DQN agent + prioritized experience replay (`DQAgent.py`, `PER.py`)
- Training entrypoint (`main.py`) and inference demo (`run.py`)
- Dockerfile for reproducible runs

## Run

Local:

- Install deps: `pip install -r requirements.txt`
- Train: `python3 main.py`
- Run a trained agent: `python3 run.py`

Docker:

- Build: `docker build -t housecleanbot:latest .`
- Train: `docker run --rm -it housecleanbot:latest python3 main.py`
- Demo: `docker run --rm -it housecleanbot:latest python3 run.py`

## Notes for Publishing

Large artifacts (weights, training frames) are intentionally gitignored. If you want to share a trained model, attach it as a GitHub Release asset or use Git LFS.

